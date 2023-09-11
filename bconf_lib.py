import os
import sys
from abc import abstractmethod
from typing import TypeVar, Generic, Optional
from typing_extensions import Self
from enum import Enum

StreamData = TypeVar('StreamData')

class Stream(Generic[StreamData]):
    """
    Base class for iterating through a stream of data.
    """
    def __init__(self):
        self.end_of_stream = False

    @abstractmethod
    def next(self) -> Optional[StreamData]:
        """
        Return the next part of the stream.
        @return None if the stream has ended.
                Otherwise, returns StreamData.
        """
        raise NotImplementedError("Stream::next() not implemented.")

    @property
    def eos(self):
        return self.end_of_stream

    def _end_of_stream(self, eos: bool):
        self.end_of_stream = eos

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> StreamData | type[StopIteration]:
        data = self.next()
        if data is None or self.end_of_stream:
            raise StopIteration
        else:
            final_data: StreamData = data
            return final_data

class InputFileStream(Stream[bytes]):
    """
    Stream data from an input file.
    """

    DEFAULT_CHUNK_SIZE = 1000

    def __init__(self, filename: str, fileptr, chunk_size):
        super().__init__()

        # `self.fptr`: File pointer to the file being read.
        # self.fptr
        # `self.chunk_size`: The max number of bytes to read from the input
        # file at a time.

        if len(filename) == 0:
            assert fileptr is not None
            self.fptr = fileptr
        else:
            if not os.path.exists(filename):
                raise FileNotFoundError("File not found: {}".format(filename))
            self.fptr = open(filename, 'rb')
        self.chunk_size = chunk_size

    @staticmethod
    def FromFilename(filename: str, chunk_size = 1000):
        return InputFileStream(filename, None, chunk_size)

    @staticmethod
    def FromFilePtr(fileptr, chunk_size = 1000):
        return InputFileStream("", fileptr, chunk_size)

    def __del__(self):
        if self.fptr is not None:
            self.fptr.close()

    def next(self) -> Optional[bytes]:
        if self.fptr is None:
            self._end_of_stream(True)
            return None
        data = self.fptr.read(self.chunk_size)
        if len(data) == 0:
            self._end_of_stream(True)
            return None
        return data

# Return the first non-whitespace/newline character in the
# string.
def firstchar(data: str) -> Optional[str]:
    index = 0
    while data[index] == ' ' or data[index] == '\n':
        index += 1
    return None if index >= len(data) else data[index]

class TokenType(Enum):
    ID = 1

class Token:
    def __init__(self, part: str, tokentype: TokenType):
        self.part = part
        self.tokentype = tokentype

    def __str__(self):
        return self.part if self.part is not None else "[None]"

    @property
    def type(self) -> TokenType:
        return self.tokentype

class TokenStream(Stream[Token]):
    """
    Iterate through tokens of an input stream.

    @note: Lines which begin with a pound are ignored.
    """
   
    # TODO: Add token groups. Right now, every word is considered
    # a token. This is not enough for a competent lexer.

    SKIPSET = set([' ', '\t'])

    def __init__(self, filestream: InputFileStream):
        super().__init__()

        # `self.buffer`: Data buffer used for storing the current chunk of the file
        # being processed.
        # `self.fstream`: The filestream for `filename`.
        # `self.ypos`: The current row position within `self.buffer`.
        # `self.xpos`: The current col position within `self.buffer`.
        self.fstream = filestream
        self.buffer: list[str] | None = None
        self.xpos = self.ypos = 0

    def next(self) -> Optional[Token]:
        if self.eos: return None
        self._refill_buffers()

        # print("Current buffer: ", self.buffer, "[x = ", self.xpos, ", y = ", self.ypos, "")
        assert (self.buffer is not None)
        if self.ypos < 0 or self.ypos >= len(self.buffer):
            self._end_of_stream(True)
            return None
     
        if self.ypos == len(self.buffer) - 1 and not self.fstream.eos:
            return self.next()

        row = self.buffer[self.ypos]
        # Start on a character outside of the skipset
        startx = self.xpos
        while startx < len(row) and row[startx] in TokenStream.SKIPSET:
            startx += 1
        if startx >= len(row):
            self.ypos += 1
            self.xpos = 0
            return self.next()

        endx = startx
        while endx < len(row) and row[endx] not in TokenStream.SKIPSET:
            endx += 1

        self.xpos = endx
        part = row[startx:endx]
        return Token(part, TokenType.ID)

    def _refill_buffers(self):
        if self.buffer is not None and self.ypos != len(self.buffer) - 1:
            return

        next_chunk = self.fstream.next()
        if next_chunk is not None: 
            # If `should_concat`, the first line of the next buffer
            # should be concatenated to the last line of the current buffer.
            # Otherwise, place them after one another.
            should_concat = True
            if next_chunk[0] == '\n':
                should_concat = False
            next_buffer = str(next_chunk).split('\n')

            if self.buffer is None:
                self.buffer = next_buffer
            else:
                # TODO: Add bounds checking for `self.buffer` and `next_buffer`.
                if should_concat:
                    self.buffer[len(self.buffer) - 1] += next_buffer[0]
                self.buffer += next_buffer[1 if should_concat else 0:]

            # Discard last `self.ypos` - 1 entries of buffer and reset buffer's
            # position.
            self.buffer = self.buffer[self.ypos:]
            self.ypos = min(len(self.buffer) - 1, 0)

def _lexer(filestream: InputFileStream):
    pass

