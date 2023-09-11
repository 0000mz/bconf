import os
import re
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
    INVALID             = 1
    ID                  = 2
    NUM                 = 3
    STRING1             = 4
    STRING2             = 5
    SQUARE_BRACKET_OPEN = 6
    SQUARE_BRACKET_CLOS = 7
    CURLY_BRACE_OPEN    = 8
    CURLY_BRACE_CLOS    = 9
    COMMA               = 10
    SEMICOLON           = 11
    COLON               = 12

def patt(patt_str: str):
    SPC_MATCH = "((\s)+)?"
    return re.compile("{}{}{}".format(SPC_MATCH, patt_str, SPC_MATCH))

kTokenMatcher = {
    TokenType.ID:                   patt("(?P<token>[a-zA-Z]+)"),
    TokenType.NUM:                  patt("(?P<token>[0-9]+)"),
    TokenType.STRING1:              patt("(?P<token>(\").*(\"))"),
    TokenType.STRING2:              patt("(?P<token>(\').*(\'))"),
    TokenType.SQUARE_BRACKET_OPEN:  patt("(?P<token>\[)"),
    TokenType.SQUARE_BRACKET_CLOS:  patt("(?P<token>\])"),
    TokenType.CURLY_BRACE_OPEN:     patt("(?P<token>\{)"),
    TokenType.CURLY_BRACE_CLOS:     patt("(?P<token>\})"),
    TokenType.COMMA:                patt("(?P<token>,)"),
    TokenType.SEMICOLON:            patt("(?P<token>;)"),
    TokenType.COLON:                patt("(?P<token>:)"),
}

class Token:
    def __init__(self):
        self.part = ""
        self.extracted_token: str | None = None
        self.extracted_token_type = TokenType.INVALID
        self.end_index = -1

    def __str__(self):
        self.compute_type()
        return self.extracted_token if self.extracted_token is not None and len(self.part) > 0 else "[None]"

    def add_part(self, part: str):
        self.part += part

    def consume_remainder(self, prev_token: Self):
        if prev_token.end_index >= 0:
            self.part = prev_token.part[prev_token.end_index:] + self.part
        self.compute_type()

    def compute_type(self) -> TokenType:
        if self.extracted_token != None:
            return self.extracted_token_type

        for token, pattern in kTokenMatcher.items():
            match = pattern.match(self.part)
            if match is not None:
                self.extracted_token = match.groupdict()["token"]
                self.extracted_token_type = token
                self.end_index = match.end('token')
        return self.extracted_token_type

    @property
    def type(self) -> TokenType:
        return self.compute_type()

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
        self.token_context = Token()

    def next(self) -> Optional[Token]:
        if self.eos: return None
        self._refill_buffers()

        # print("Current buffer: ", self.buffer, "[x = ", self.xpos, ", y = ", self.ypos, "")
        assert (self.buffer is not None)
        if self.ypos < 0 or self.ypos >= len(self.buffer):
            # Process remaining values in current token
            self.token_context.compute_type()
            if self.token_context.type == TokenType.INVALID:
                self._end_of_stream(True)
                return None
            next_token = self.token_context
            self.token_context = Token()
            self.token_context.consume_remainder(next_token)
            return next_token
     
        if self.ypos == len(self.buffer) - 1 and not self.fstream.eos:
            return self.next()

        row = self.buffer[self.ypos]
        # Start on a character outside of the skipset
        prestartx = self.xpos
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
        part = row[prestartx:endx]
        self.token_context.add_part(part)

        if self.token_context.type == TokenType.INVALID:
            return self.next()

        next_token = self.token_context
        self.token_context = Token()
        self.token_context.consume_remainder(next_token)
        return next_token

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

