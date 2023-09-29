import os
import re
import sys
import copy
import logging
import random
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

        self.fptr = None
        if len(filename) == 0:
            assert fileptr is not None
            self.fptr = fileptr
        else:
            if not os.path.exists(filename):
                raise FileNotFoundError("File not found: {}".format(filename))
            self.fptr = open(filename, 'r')
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
        return []

kTokenMap = {
        'A': TokenType.ID,
        'B': TokenType.NUM,
        'C': TokenType.HEX,
        'D': TokenType.STRING1,
        'E': TokenType.STRING2,
        'F': TokenType.SQUARE_BRACKET_OPEN,
        'G': TokenType.SQUARE_BRACKET_CLOS,
        'H': TokenType.CURLY_BRACE_OPEN,
        'I': TokenType.CURLY_BRACE_CLOS,
        'J': TokenType.COMMA,
        'K': TokenType.SEMICOLON,
        'L': TokenType.COLON,
        'M': TokenType.EQUAL,
}
kTokenMapReverse = dict(map(lambda el: (el[1], el[0]), kTokenMap.items()))

class BConfGrammar(Grammar):

    def get(self):
        return [
            # identifier { }
            "AHI",
        ]

_QUALIFIER_STARTS   = set(['+', '*', '?', '{'])
_QUALIFIER_END      = set(['+', '*', '?', '}'])
def is_beginning_of_qualifier(el: str) -> bool:
    return el in _QUALIFIER_STARTS
def is_end_of_qualifier(el: str) -> bool:
    return el in _QUALIFIER_END

def split_regex_into_groups(grammar_regex: str) -> list[str]:
    groups: list[str] = []
    current_complex: Optional[str] = None
    nested_paren_ct = 0

    def extract_qualifier(regex: str) -> str:
        assert len(regex) > 0
        if regex[0] == '{':
            index = 1
            while index < len(regex) and regex[index] != '}':
                index += 1
            assert regex[index] == '}'
            return regex[:index+1]
        else:
            return regex[0]

    def add_simple_element(el: str):
        nonlocal current_complex, groups
        if current_complex is not None:
            current_complex += el
        else:
            groups.append(el)

    i = 0
    while i < len(grammar_regex):
        el = grammar_regex[i]
        if el == '(':
            nested_paren_ct += 1
        elif el == ')':
            nested_paren_ct -= 1

        if el == '(' and nested_paren_ct == 1:
            current_complex = '('
        elif el == ')' and nested_paren_ct == 0:
            assert current_complex is not None
            current_complex += ')'
            next_el = grammar_regex[i+1] if i+1 < len(grammar_regex) else None
            if next_el is not None and is_beginning_of_qualifier(next_el):
                qualifier = extract_qualifier(grammar_regex[i+1:])
                current_complex += qualifier
                i += len(qualifier)
            groups.append(current_complex)
            current_complex = None

        else:
            el_to_add = el
            next_el = grammar_regex[i+1] if i+1 < len(grammar_regex) else None
            if next_el is not None and is_beginning_of_qualifier(next_el):
                qualifier = extract_qualifier(grammar_regex[i+1:])
                el_to_add += qualifier
                i += len(qualifier)
            add_simple_element(el_to_add)
        i += 1
    return groups

class Qualifier(Enum):
    PLUS                 = ord('+')
    OPTIONAL             = ord('?')
    MULTIPLIER           = ord('*')
    CONSTANT_REPLICATION = ord('{') # {2}

def extract_qualifier_from_end(grammar: str) -> Optional[tuple[Qualifier, str]]:
    if len(grammar) == 0 or not is_end_of_qualifier(grammar[-1]):
        return None
    start = len(grammar) - 1
    if grammar[-1] == '}':
        while start >= 0 and grammar[start] != '{':
            start -= 1
    qualifier_str = grammar[start:]
    if qualifier_str[0] == '{':
        return (Qualifier.CONSTANT_REPLICATION, qualifier_str)
    elif qualifier_str[0] == '+':
        return (Qualifier.PLUS, qualifier_str)
    elif qualifier_str[0] == '*':
        return (Qualifier.MULTIPLIER, qualifier_str)
    elif qualifier_str[0] == '?':
        return (Qualifier.OPTIONAL, qualifier_str)
    raise TypeError("Qualifier not defined.")

def deparenthesize(grammar: str) -> str:
    if len(grammar) > 0 and grammar[0] == '(' and grammar[-1] == ')':
         return grammar[1:-1]
    return grammar

def expand_grammar(grammar: str) -> list[str]:
    """
    Given a complex grammar, generate all 1 level expansions of the complex grammar.

    A grammar can be expanded if it is parenthesized and/or it has a qualifier.
    """
    def _get_grammar_and_qualifier(grammar: str) -> tuple[str, Optional[tuple[str, Qualifier]]]:
        """
        Get the grammar and qualifier, separate from each other.
        i.e. (AB)+ -> (AB, (+, PLUS))
        Deparenthesize the subject befor returning.
        """
        qualifier_info = extract_qualifier_from_end(grammar)
        if qualifier_info is None:
            return (deparenthesize(grammar), None)
        [qualifier_enum, qualifier_value] = qualifier_info
        return (deparenthesize(grammar[:len(grammar)-len(qualifier_value)]), (qualifier_value, qualifier_enum))

    def _extract_constant_replication(const_qualifier: str) -> int:
        assert len(const_qualifier) > 0
        assert const_qualifier[0] == '{'
        assert const_qualifier[-1] == '}'
        return int(const_qualifier[1:-1])

    def _apply_qualifier(grammar: str, qualifier: Qualifier, qualifier_str: str) -> list[str]:
        match qualifier:
            case Qualifier.PLUS:
                return [grammar, f'{grammar}({grammar})+']
            case Qualifier.OPTIONAL:
                return ['', grammar]
            case Qualifier.MULTIPLIER:
                return ['', grammar, f'{grammar}({grammar})*']
            case CONSTANT_REPLICATION:
                return [''.join(grammar for i in range(_extract_constant_replication(qualifier_str)))]

    [stripped_grammar, qualifier] = _get_grammar_and_qualifier(grammar)
    if qualifier is None:
        return [stripped_grammar]

    [qualifier_str, qualifier_enum] = qualifier
    return _apply_qualifier(stripped_grammar, qualifier_enum, qualifier_str)

_TOKENLST = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t',
             'u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
def random_id(length: int) -> str:
    return ''.join([_TOKENLST[random.randint(0, sys.maxsize) % len(_TOKENLST)] for _ in range(length)])

class GrammarNode:

    def __init__(self, token_id: str | TokenGroupType):
        self.token_match: Optional[str] = None
        self.group_type: Optional[TokenGroupType] = None
        self._id = random_id(25)
        self._token_index: int = -1
        self._valid: bool = True
        logging.debug("GrammarNode id: %s", self._id)

        if isinstance(token_id, TokenGroupType):
            self.group_type = token_id
        else:
            self.token_match = token_id
        self._children: list[Self] = []
        self._parent: Optional[Self] = None
  
    def __str__(self):
        if self.group_type is not None:
            return "[GROUP_TYPE]" # TODO: Print the actual group type info
        return self.token_match
     
    @property
    def token_index(self) -> int:
        return self._token_index

    @property
    def id(self) -> str:
        return self._id

    @property
    def data(self) -> str | TokenGroupType:
        assert self.group_type is not None or self.token_match is not None
        return self.group_type if self.group_type is not None else self.token_match # type: ignore

    @property
    def children(self) -> list[Self]:
        return self._children
    
    @property
    def parent(self) -> Optional[Self]:
        return self._parent

    @property
    def valid(self) -> bool:
        return self._valid

    def mark_invalid(self):
        self._valid = False

    def match_token(self, token: Token) -> bool:
        ttype: TokenType = token.compute_type()
        if self.group_type is not None:
            # TODO: Implement match for group type
            return False
        elif self.is_complex():
            # TODO: Implement match for complex type
            return False
        else:
            return ttype in kTokenMapReverse and self.token_match == kTokenMapReverse[ttype]

    # Keep track of the token index that is associated with this grammar node.
    def attach_token_index(self, index: int):
        self._token_index = index

    # A complex grammar node is a grammar node that has an expansion.
    # i.e. A+ has an expansion to AA+
    # When exploring complex grammar nodes, they need to be expanded in order to
    # match tokens to them.
    def is_complex(self) -> bool:
        if self.token_match is None:
            return False
        return len(self.token_match) > 1 or not self.token_match.isalpha()

    def is_same_node_type(self, node: Self) -> bool:
        if self.token_match is not None and node.token_match is not None:
            return self.token_match == node.token_match
        else:
            return self.group_type == node.group_type

    def set_parent(self, node: Optional[Self]):
        self._parent = node

    # Add a node as a child of this node. Return the added node.
    def add_child_node(self, grammar_node: Self) -> Self:
        for child_node in self._children:
            if child_node.is_same_node_type(grammar_node):
                return child_node
        self._children.append(grammar_node)
        grammar_node.set_parent(self)
        return self._children[len(self._children) - 1]

    # Return a copy of the current node, excluding the children.
    def clone(self) -> Self:
        new_node = GrammarNode(self.group_type if self.group_type is not None else self.token_match) # type: ignore
        return new_node # type: ignore

    # Given a node, copy all the children and the children's childrens recursively until the
    # entire subtree of `src` is copied to the current grammar tree.
    def copy_children_subtrees(self, src: Self):
        src_stack: list[GrammarNode] = [src]
        dst_stack: list[GrammarNode] = [self]

        while src_stack:
            next_src = src_stack.pop()
            next_dst = dst_stack.pop()

            for src_child in next_src.children:
                dst_child = src_child.clone()
                next_dst.add_child_node(dst_child)
                src_stack.append(src_child)
                dst_stack.append(dst_child)

    # Given that the current node is complex, create separate grammar trees for each possible
    # expansion of this complex node. Return the roots of these generated grammar trees.
    def generate_expansions(self) -> list[Self]:
        assert self.is_complex()

        logging.debug("[generate_expansions] token_match = %s", self.token_match)
        assert self.token_match is not None
        expansions = expand_grammar(self.token_match)
        expansion_nodes: list[Self] = []

        # Create the node and duplicate the children of the current node to the
        # created node.
        for expansion in expansions:
            new_node = GrammarNode(expansion)
            new_node.copy_children_subtrees(self)
            expansion_nodes.append(new_node) # type: ignore

        return expansion_nodes

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
        for grammar_regex in self.grammar:
            current_node = self._root
            for grammar_regex_part in split_regex_into_groups(grammar_regex):
                current_node = current_node.add_child_node(GrammarNode(grammar_regex_part))
            current_node.add_child_node(GrammarNode(TokenGroupType.END))
        return True

# Return true if the token matches the grammar for the given `index`.
def matches_grammar(grammar: list[TokenType], token: Token, index: int):
    return index < len(grammar) and grammar[index] == token.type

class Parser:

    def __init__(self, filestream: InputFileStream):
        assert(filestream is not None)
        self.tokenstream = TokenStream(filestream)
        self.parsed_data: dict[str, Any] = {}
        self._tokens: list[Token] = []
        self.grammar_tree: Optional[GrammarTree] = None
    
    # Cache the next token internally from the token stream.
    # If no more tokens are available, return False.
    def _cache_next_token(self) -> bool:
        next_token = self.tokenstream.next()
        logging.debug("next token = %s", str(next_token))
        if next_token is None:
            return False
        self._tokens.append(next_token)
        return True

    # Get the `index`th token from the token stream
    def _get_token(self, index: int) -> Optional[Token]:
        assert index >= 0
        while index >= len(self._tokens):
            if not self._cache_next_token():
                return None
        return self._tokens[index]

    def parse(self) -> bool:
        grammar_tree = GrammarTree.InitFromGrammar(BConfGrammar())
        # The stack holds entries of type: tuple[GrammarNode, int | None]
        # The GrammarNode is the node to be explored and the int is the
        # index of the token that should be resolved by that node.
        #
        # Special case: root node must satisfy token index -1, a non-existent
        # node. This should be skipped bc the root node here is the START token.
        grammar_stack: list[tuple[GrammarNode, int]] = [(grammar_tree.root, -1)]
        logging.debug("Initial grammar stack for parsing: %s", grammar_stack)
        start_node = True

        # Try to find a path in the grammar tree that satisfies
        # the entire sequence of tokens in the token stream.
        # Expand complex nodes if necessary.
        while grammar_stack:
            [current_node, token_index] = grammar_stack.pop()

            skip_token = False
            if token_index >= 0:
                # Force the population of the token lst.
                token = self._get_token(token_index)
                if token is not None:
                    logging.debug(
                            "Checking if %s matches %s [%s]",
                            str(current_node),
                            str(token),
                            kTokenMapReverse[token.compute_type()]
                            if token.compute_type() in kTokenMapReverse 
                            else "[Unknown]"
                    )
                    skip_token = not current_node.match_token(token)
                    logging.debug("match = %s", not skip_token)

            logging.debug(
                "parsing next node: %s (token index = %s)",
                current_node,
                token_index
            )
            current_node.attach_token_index(token_index)

            # Complex nodes should be expanded.
            if current_node.is_complex():
                logging.debug("Expanding complex.")
                expansion_nodes = current_node.generate_expansions()
                # Attach each expansion as a sibbling of the `current_node`
                for expansion_node in expansion_nodes:
                    assert current_node.parent is not None
                    current_node.parent.add_child_node(expansion_node)
                    grammar_stack.append((current_node, token_index))
            
            if not start_node and skip_token:
                current_node.mark_invalid()

            if not skip_token:
                for child_node in current_node.children:
                    # TODO: (OPT) Add children to stack in order of least complex to
                    # most for optimization.
                    grammar_stack.append((child_node, token_index + 1))
            
            if start_node: start_node = False

        self.grammar_tree = grammar_tree
        return False

    @property
    def tokens(self) -> list[Token]:
        return self._tokens
    
    @property
    def data(self) -> dict[str, Any]:
        return self.parsed_data

    @property
    def tree(self) -> Optional[GrammarTree]:
        return self.grammar_tree

def _lexer(filestream: InputFileStream):
    pass

