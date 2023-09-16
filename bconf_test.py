import io
import unittest
import bconf_lib
from bconf_lib import InputFileStream, TokenStream, TokenType, Parser
from bconf_lib import Grammar, GrammarTree, TokenGroupType
from parameterized import parameterized # type: ignore

class InputFileStreamTest(unittest.TestCase):

    def test_read_memfile(self):
        memfile_data = "This is a memfile"
        memfile = io.StringIO(memfile_data)
        stream = InputFileStream.FromFilePtr(memfile)

        bytes_read = 0
        for data in stream:
            bytes_read += len(data)
        self.assertEqual(bytes_read, len(memfile_data))

class TokenStreamTest(unittest.TestCase):
    
    @parameterized.expand([
        # With chunk size of 5 and newline at the end of first chunk, test that
        # the tokenization is correct.
        ("This\nis a bounds check", ["This", "is", "a", "bounds", "check"], 5),
        # With chunk size of 4 and newline at the start of second chunk, test that
        # the tokenization is correct.
        ("This\nis a bounds check", ["This", "is", "a", "bounds", "check"], 4),
        # With the chunk size of 2, test that the chunk 1, chunk 2, and chunk 3 
        # correctly concatenate to a full token.
        ("Thisisanothertest end", ["Thisisanothertest", "end"], 2),

        ("This is a message", ["This", "is", "a", "message"], 4),
        ("""


        This message is filled with multiple newlines


         """, ["This", "message", "is", "filled", "with", "multiple", "newlines"], 4),
    ])
    def test_parse_id_tokens(self, filecontents: str, tokenlst: list[str], chunksize: int):
        memfile = io.StringIO(filecontents)

        fstream = InputFileStream.FromFilePtr(memfile, chunksize)
        tokstream = TokenStream(fstream)
        index = 0
        for token in tokstream:
            self.assertLess(index, len(tokenlst))
            expected_token = tokenlst[index]
            self.assertEqual(str(token), expected_token)
            self.assertEqual(token.type, TokenType.ID) # type: ignore
            index += 1
        self.assertEqual(index, len(tokenlst))

    @parameterized.expand([
        (("identifier 2301"), [("identifier", TokenType.ID), ("2301", TokenType.NUM)]),
        (
            ("""identifier "This is a string 1232" identifier"""), 
            [
                ("identifier", TokenType.ID),
                ('"This is a string 1232"', TokenType.STRING1),
                ("identifier", TokenType.ID)
            ]
        ),
        (
            ("""identifier "string captures nested string \"test\"" identifier"""), 
            [
                ("identifier", TokenType.ID),
                ('"string captures nested string \"test\""', TokenType.STRING1),
                ("identifier", TokenType.ID)
            ]
        ),
        (
            ("""identifier 'This is a string 1232' identifier"""), 
            [
                ("identifier", TokenType.ID),
                ("'This is a string 1232'", TokenType.STRING2),
                ("identifier", TokenType.ID)
            ]
        ),
        (
            ("""identifier 'string captures nested string \'test\'' identifier"""), 
            [
                ("identifier", TokenType.ID),
                ("'string captures nested string \'test\''", TokenType.STRING2),
                ("identifier", TokenType.ID)
            ]
        ),
        (
            ("""array[]"""),
            [
                ("array", TokenType.ID),
                ("[", TokenType.SQUARE_BRACKET_OPEN),
                ("]", TokenType.SQUARE_BRACKET_CLOS)
            ]
        ),
        (
            ("""array [ ]"""),
            [
                ("array", TokenType.ID),
                ("[", TokenType.SQUARE_BRACKET_OPEN),
                ("]", TokenType.SQUARE_BRACKET_CLOS)
            ]
        ),
        (
            ("""brace{}"""),
            [
                ("brace", TokenType.ID),
                ("{", TokenType.CURLY_BRACE_OPEN),
                ("}", TokenType.CURLY_BRACE_CLOS)
            ]
        ),
        (
            ("""brace { }"""),
            [
                ("brace", TokenType.ID),
                ("{", TokenType.CURLY_BRACE_OPEN),
                ("}", TokenType.CURLY_BRACE_CLOS)
            ]
        ),
        (
            ("""[identifier,identifier,]"""),
            [
                ("[", TokenType.SQUARE_BRACKET_OPEN),
                ("identifier", TokenType.ID),
                (",", TokenType.COMMA),
                ("identifier", TokenType.ID),
                (",", TokenType.COMMA),
                ("]", TokenType.SQUARE_BRACKET_CLOS)
            ]
        ),
        (
            ("""[ identifier , identifier , ] """),
            [
                ("[", TokenType.SQUARE_BRACKET_OPEN),
                ("identifier", TokenType.ID),
                (",", TokenType.COMMA),
                ("identifier", TokenType.ID),
                (",", TokenType.COMMA),
                ("]", TokenType.SQUARE_BRACKET_CLOS)
            ]
        ),
        (
            ("""semicolon; another;"""),
            [
                ("semicolon", TokenType.ID),
                (";", TokenType.SEMICOLON),
                ("another", TokenType.ID),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""key:value;"""),
            [
                ("key", TokenType.ID),
                (":", TokenType.COLON),
                ("value", TokenType.ID),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""key : value;"""),
            [
                ("key", TokenType.ID),
                (":", TokenType.COLON),
                ("value", TokenType.ID),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""key=value;"""),
            [
                ("key", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("value", TokenType.ID),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""key = value;"""),
            [
                ("key", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("value", TokenType.ID),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""number = 123.456;"""),
            [
                ("number", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("123.456", TokenType.NUM),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""number = 123.;"""),
            [
                ("number", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("123.", TokenType.NUM),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""number = -123.;"""),
            [
                ("number", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("-123.", TokenType.NUM),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""number = 0xff;"""),
            [
                ("number", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("0xff", TokenType.HEX),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""number = 0xffffffffffff;"""),
            [
                ("number", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("0xffffffffffff", TokenType.HEX),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""number = -0x0023f9af12;"""),
            [
                ("number", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("-0x0023f9af12", TokenType.HEX),
                (";", TokenType.SEMICOLON),
            ]
        ),
        (
            ("""
             # This is a comment
             key = value;
             # This is another comment
             key:value;
             """),
            [
                ("key", TokenType.ID),
                ("=", TokenType.EQUAL),
                ("value", TokenType.ID),
                (";", TokenType.SEMICOLON),
                ("key", TokenType.ID),
                (":", TokenType.COLON),
                ("value", TokenType.ID),
                (";", TokenType.SEMICOLON),
            ]
        )
    ])
    def test_parse_all_tokens(self, filecontents: str, tokenlst: list[tuple[str, TokenType]]):
        memfile = io.StringIO(filecontents)

        fstream = InputFileStream.FromFilePtr(memfile, 4)
        tokstream = TokenStream(fstream)
        index = 0
        for token in tokstream:
            self.assertLess(index, len(tokenlst))
            expected_token = tokenlst[index]
            self.assertEqual(str(token), expected_token[0])
            self.assertEqual(token.type, expected_token[1]) # type: ignore
            index += 1
        self.assertEqual(index, len(tokenlst))

    def test_eof_with_unidentified_token(self):
        contents = """
        "This is a string that is unterminated
        """
        memfile = io.StringIO(contents)

        fstream = InputFileStream.FromFilePtr(memfile, 4)
        tokstream = TokenStream(fstream)

        token = tokstream.next()
        self.assertIsNotNone(token)
        self.assertEqual(token.type, TokenType.INVALID)
        self.assertEqual(str(token), """        "This is a string that is unterminated""")

        token = tokstream.next()
        self.assertIsNone(token)

class TestGrammar(Grammar):
    def get(self):
            return [
                    'ABCD',
            ]

class TestGrammar2(Grammar):
    def get(self):
            return [
                    'ABCD',
                    'AXYZ',
            ]


class GrammarTreeTest(unittest.TestCase):
    
    def test_grammar_1_tree_gen(self):
        gtree = GrammarTree.InitFromGrammar(TestGrammar())
        self.assertEqual(gtree.root.data, TokenGroupType.START)

        # START
        #   |
        #   A
        #   |
        #   B
        #   |
        #   C
        #   |
        #   D
        #   |
        #  END

        # START -> A
        children = gtree.root.children
        self.assertEqual(1, len(children))
        self.assertEqual('A', children[0].data)
        
        # A -> B
        children = children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual('B', children[0].data)

        # B -> C
        children = children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual('C', children[0].data)

        # C -> D
        children = children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual('D', children[0].data)
        
        # D -> END
        children = children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual(TokenGroupType.END, children[0].data)

        # END -> None
        children = children[0].children
        self.assertEqual(0, len(children))        

    def test_grammar_2_tree_gen(self):
        gtree = GrammarTree.InitFromGrammar(TestGrammar2())
        self.assertEqual(gtree.root.data, TokenGroupType.START)

        # START
        #   |
        #   A
        #   |\
        #   | \
        #   B  X
        #   |  |
        #   C  Y
        #   |  |
        #   D  Z
        #   |  |
        #  END END

        # START -> A
        children = gtree.root.children
        self.assertEqual(1, len(children))
        self.assertEqual('A', children[0].data)
        
        # A -> [B, X]
        mid_children = children[0].children
        self.assertEqual(2, len(mid_children))
        self.assertEqual('B', mid_children[0].data)
        self.assertEqual('X', mid_children[1].data)

        # B -> C
        children = mid_children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual('C', children[0].data)

        # C -> D
        children = children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual('D', children[0].data)
        
        # D -> END
        children = children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual(TokenGroupType.END, children[0].data)

        # END -> None
        children = children[0].children
        self.assertEqual(0, len(children))        
        
        # X -> Y
        children = mid_children[1].children
        self.assertEqual(1, len(children))
        self.assertEqual('Y', children[0].data)

        # Y -> Z
        children = children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual('Z', children[0].data)

        # Z -> END
        children = children[0].children
        self.assertEqual(1, len(children))
        self.assertEqual(TokenGroupType.END, children[0].data)

        # END -> None
        children = children[0].children
        self.assertEqual(0, len(children))

class RegexSplitTest(unittest.TestCase):

    @parameterized.expand([
        ('', []),
        ('A', ['A']),
        ('AA', ['A', 'A']),
        ('A(AA)A', ['A', '(AA)', 'A']),
        ('A(AA)+A', ['A', '(AA)+', 'A']),
        ('A+', ['A+']),
        ('AA+', ['A', 'A+']),
        ('A?', ['A?']),
        ('A*', ['A*']),
        ('A{2}', ['A{2}']),
        ('ABCDE(WXYZ)?LMN', ['A', 'B', 'C', 'D', 'E', '(WXYZ)?', 'L', 'M', 'N']),
        ('ABCDE(WXYZ){2}LMN', ['A', 'B', 'C', 'D', 'E', '(WXYZ){2}', 'L', 'M', 'N']),
        ('A(ABC(DEF)GH)I', ['A', '(ABC(DEF)GH)', 'I']),
        ('Z(A(A(A(A(A(A(A(A))))))))Z', ['Z', '(A(A(A(A(A(A(A(A))))))))', 'Z']),
        ('Z(A(A(A(A(A(A(A(A)))))))){1}Z', ['Z', '(A(A(A(A(A(A(A(A)))))))){1}', 'Z']),
        ('Z(A(A(A(A(A(A(A(A))))))))*Z', ['Z', '(A(A(A(A(A(A(A(A))))))))*', 'Z']),
        ('Z(A(A(A(A(A(A(A(A))))))))?Z', ['Z', '(A(A(A(A(A(A(A(A))))))))?', 'Z']),
        ('Z(A(A(A(A(A(A(A(A)))+)))))?Z', ['Z', '(A(A(A(A(A(A(A(A)))+)))))?', 'Z']),
    ])
    def test_regex_split(self, pattern, expected_groups):

        groups = bconf_lib.split_regex_into_groups(pattern)
        self.assertEqual(
            len(expected_groups),
            len(groups),
            "expected {} vs actual {}".format(expected_groups, groups)
        )
        for i in range(len(groups)):
            self.assertEqual(groups[i], expected_groups[i])

# class ParserTest(unittest.TestCase):
#
#     # TODO: Impl
#     def test_parser_test(self):
#         contents = """
#
#         identifier {
#
#         }
#
#         """
#         memfile = io.StringIO(contents)
#         parser = Parser(InputFileStream.FromFilePtr(memfile))
#
#         self.assertTrue(parser.parse())
#         # self.assertTrue("identifier" in parser.data)
#         # self.assertEqual(parser.data["identifier"], {})


if __name__ == '__main__':
    unittest.main()

