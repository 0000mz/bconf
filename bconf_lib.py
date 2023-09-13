import os
import re
import sys
import copy
from abc import abstractmethod
from typing import TypeVar, Generic, Optional, Any
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
    while index < len(data) and (data[index] == ' ' or data[index] == '\n'):
        index += 1
    return None if index >= len(data) else data[index]

class TokenType(Enum):
    INVALID             = 1
    ID                  = 2
    NUM                 = 3
    HEX                 = 4
    STRING1             = 5
    STRING2             = 6
    SQUARE_BRACKET_OPEN = 7
    SQUARE_BRACKET_CLOS = 8
    CURLY_BRACE_OPEN    = 9
    CURLY_BRACE_CLOS    = 10
    COMMA               = 11
    SEMICOLON           = 12
    COLON               = 13
    EQUAL               = 14

def patt(patt_str: str):
    SPC_MATCH = "((\s)+)?"
    return re.compile("{}{}{}".format(SPC_MATCH, patt_str, SPC_MATCH))

kTokenMatcher = {
    TokenType.ID:                   patt("(?P<token>[a-zA-Z]+)"),
    TokenType.NUM:                  patt("(?P<token>(-)?[0-9]+((\.)([0-9]+)?)?)"),
    TokenType.HEX:                  patt("(?P<token>(-)?(0x)[abcdef0-9]+)"),
    TokenType.STRING1:              patt("(?P<token>(\").*(\"))"),
    TokenType.STRING2:              patt("(?P<token>(\').*(\'))"),
    TokenType.SQUARE_BRACKET_OPEN:  patt("(?P<token>\[)"),
    TokenType.SQUARE_BRACKET_CLOS:  patt("(?P<token>\])"),
    TokenType.CURLY_BRACE_OPEN:     patt("(?P<token>\{)"),
    TokenType.CURLY_BRACE_CLOS:     patt("(?P<token>\})"),
    TokenType.COMMA:                patt("(?P<token>,)"),
    TokenType.SEMICOLON:            patt("(?P<token>;)"),
    TokenType.COLON:                patt("(?P<token>:)"),
    TokenType.EQUAL:                patt("(?P<token>=)"),
}

class Token:
    def __init__(self):
        self.part = ""
        self.extracted_token: str | None = None
        self.extracted_token_type = TokenType.INVALID
        self.end_index = -1

    def __str__(self):
        self.compute_type()
        if self.extracted_token is not None:
            return self.extracted_token
        elif len(self.part) > 0:
            return self.part
        return "[None]"

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

        assert (self.buffer is not None)
        if self.ypos < 0 or self.ypos >= len(self.buffer):
            # Process remaining values in current token
            self.token_context.compute_type()
            if self.token_context.type == TokenType.INVALID:
                self._end_of_stream(True)
                return self._reset_token()
            return self._reset_token()
     
        if self.ypos == len(self.buffer) - 1 and not self.fstream.eos:
            return self.next()

        row = self.buffer[self.ypos]
        while firstchar(row) == '#':
            self.ypos += 1
            self.xpos = 0
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

        return self._reset_token()

    def _reset_token(self):
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

class TokenGroupType(Enum):
    START = 0
    END   = 1

class Grammar:

    def get(self):
        return [
            [ TokenType.ID, TokenType.CURLY_BRACE_OPEN, TokenType.CURLY_BRACE_CLOS ]
        ]

class GrammarNode:

    def __init__(self, token_type: TokenType | TokenGroupType):
        self.token_type: Optional[TokenType] = None
        self.group_type: Optional[TokenGroupType] = None

        if isinstance(token_type, TokenGroupType):
            self.group_type = token_type
        else:
            self.token_type = token_type

        self._children: list[Self] = []
    
    @property
    def token(self) -> TokenType | TokenGroupType:
        return self.token_type if self.token_type is not None else self.group_type

    @property
    def children(self) -> list[Self]:
        return self._children

    # Add a node as a child of this node. Return the added node.
    def add_node(self, grammar_node: Self) -> Self:
        for child_node in self._children:
            if grammar_node.token == child_node.token:
                return child_node
        
        self._children.append(grammar_node)
        return self._children[len(self._children) - 1]

class GrammarTree:

    def __init__(self, grammar: Grammar):
        self.grammar = grammar.get()
        self._root = GrammarNode(TokenGroupType.START)
        assert self._generate_grammar_tree()

    @staticmethod
    def InitFromGrammar(grammar: Grammar):
        return GrammarTree(grammar)
   
    @property
    def root(self) -> GrammarNode:
        return self._root

    def _generate_grammar_tree(self) -> bool:
        for grammar_group in self.grammar:
            current_node = self._root
            for grammar_token in grammar_group:
                current_node = current_node.add_node(GrammarNode(grammar_token))
            current_node.add_node(GrammarNode(TokenGroupType.END))
        return True

# Return true if the token matches the grammar for the given `index`.
def matches_grammar(grammar: list[TokenType], token: Token, index: int):
    return index < len(grammar) and grammar[index] == token.type

class Parser:

    def __init__(self, filestream: InputFileStream):
        assert(filestream is not None)
        self.tokenstream = TokenStream(filestream)
        self.parsed_data: dict[str, Any] = {}

    def parse(self) -> bool:

        index = 0
        current_grammar = copy.deepcopy(Grammer().get())
        for token in self.tokenstream:
            if token.type == TokenType.INVALID:
                # TODO: Return the actual error in the payload.
                raise SyntaxError("File does not match bconf grammar.")
            
            current_grammer = list(
                    filter(
                        lambda el: matches_grammar(current_grammar, token, index), current_grammar
                    )
                )
            index += 1
        return len(current_grammar) == 1 and index == len(current_grammar[0])

    @property
    def data(self) -> dict[str, Any]:
        return self.parsed_data

def _lexer(filestream: InputFileStream):
    pass

