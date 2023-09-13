import io
import unittest
from enum import Enum
from bconf_lib import InputFileStream, TokenStream, TokenType, Parser, Grammar, GrammarTree, TokenGroupType 
from parameterized import parameterized # type: ignore

# class InputFileStreamTest(unittest.TestCase):
#
#     def test_read_memfile(self):
#         memfile_data = "This is a memfile"
#         memfile = io.StringIO(memfile_data)
#         stream = InputFileStream.FromFilePtr(memfile)
#
#         bytes_read = 0
#         for data in stream:
#             bytes_read += len(data)
#         self.assertEqual(bytes_read, len(memfile_data))
#
# class TokenStreamTest(unittest.TestCase):
#     
#     @parameterized.expand([
#         # With chunk size of 5 and newline at the end of first chunk, test that
#         # the tokenization is correct.
#         ("This\nis a bounds check", ["This", "is", "a", "bounds", "check"], 5),
#         # With chunk size of 4 and newline at the start of second chunk, test that
#         # the tokenization is correct.
#         ("This\nis a bounds check", ["This", "is", "a", "bounds", "check"], 4),
#         # With the chunk size of 2, test that the chunk 1, chunk 2, and chunk 3 
#         # correctly concatenate to a full token.
#         ("Thisisanothertest end", ["Thisisanothertest", "end"], 2),
#
#         ("This is a message", ["This", "is", "a", "message"], 4),
#         ("""
#
#
#         This message is filled with multiple newlines
#
#
#          """, ["This", "message", "is", "filled", "with", "multiple", "newlines"], 4),
#     ])
#     def test_parse_id_tokens(self, filecontents: str, tokenlst: list[str], chunksize: int):
#         memfile = io.StringIO(filecontents)
#
#         fstream = InputFileStream.FromFilePtr(memfile, chunksize)
#         tokstream = TokenStream(fstream)
#         index = 0
#         for token in tokstream:
#             self.assertLess(index, len(tokenlst))
#             expected_token = tokenlst[index]
#             self.assertEqual(str(token), expected_token)
#             self.assertEqual(token.type, TokenType.ID) # type: ignore
#             index += 1
#         self.assertEqual(index, len(tokenlst))
#
#     @parameterized.expand([
#         (("identifier 2301"), [("identifier", TokenType.ID), ("2301", TokenType.NUM)]),
#         (
#             ("""identifier "This is a string 1232" identifier"""), 
#             [
#                 ("identifier", TokenType.ID),
#                 ('"This is a string 1232"', TokenType.STRING1),
#                 ("identifier", TokenType.ID)
#             ]
#         ),
#         (
#             ("""identifier "string captures nested string \"test\"" identifier"""), 
#             [
#                 ("identifier", TokenType.ID),
#                 ('"string captures nested string \"test\""', TokenType.STRING1),
#                 ("identifier", TokenType.ID)
#             ]
#         ),
#         (
#             ("""identifier 'This is a string 1232' identifier"""), 
#             [
#                 ("identifier", TokenType.ID),
#                 ("'This is a string 1232'", TokenType.STRING2),
#                 ("identifier", TokenType.ID)
#             ]
#         ),
#         (
#             ("""identifier 'string captures nested string \'test\'' identifier"""), 
#             [
#                 ("identifier", TokenType.ID),
#                 ("'string captures nested string \'test\''", TokenType.STRING2),
#                 ("identifier", TokenType.ID)
#             ]
#         ),
#         (
#             ("""array[]"""),
#             [
#                 ("array", TokenType.ID),
#                 ("[", TokenType.SQUARE_BRACKET_OPEN),
#                 ("]", TokenType.SQUARE_BRACKET_CLOS)
#             ]
#         ),
#         (
#             ("""array [ ]"""),
#             [
#                 ("array", TokenType.ID),
#                 ("[", TokenType.SQUARE_BRACKET_OPEN),
#                 ("]", TokenType.SQUARE_BRACKET_CLOS)
#             ]
#         ),
#         (
#             ("""brace{}"""),
#             [
#                 ("brace", TokenType.ID),
#                 ("{", TokenType.CURLY_BRACE_OPEN),
#                 ("}", TokenType.CURLY_BRACE_CLOS)
#             ]
#         ),
#         (
#             ("""brace { }"""),
#             [
#                 ("brace", TokenType.ID),
#                 ("{", TokenType.CURLY_BRACE_OPEN),
#                 ("}", TokenType.CURLY_BRACE_CLOS)
#             ]
#         ),
#         (
#             ("""[identifier,identifier,]"""),
#             [
#                 ("[", TokenType.SQUARE_BRACKET_OPEN),
#                 ("identifier", TokenType.ID),
#                 (",", TokenType.COMMA),
#                 ("identifier", TokenType.ID),
#                 (",", TokenType.COMMA),
#                 ("]", TokenType.SQUARE_BRACKET_CLOS)
#             ]
#         ),
#         (
#             ("""[ identifier , identifier , ] """),
#             [
#                 ("[", TokenType.SQUARE_BRACKET_OPEN),
#                 ("identifier", TokenType.ID),
#                 (",", TokenType.COMMA),
#                 ("identifier", TokenType.ID),
#                 (",", TokenType.COMMA),
#                 ("]", TokenType.SQUARE_BRACKET_CLOS)
#             ]
#         ),
#         (
#             ("""semicolon; another;"""),
#             [
#                 ("semicolon", TokenType.ID),
#                 (";", TokenType.SEMICOLON),
#                 ("another", TokenType.ID),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""key:value;"""),
#             [
#                 ("key", TokenType.ID),
#                 (":", TokenType.COLON),
#                 ("value", TokenType.ID),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""key : value;"""),
#             [
#                 ("key", TokenType.ID),
#                 (":", TokenType.COLON),
#                 ("value", TokenType.ID),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""key=value;"""),
#             [
#                 ("key", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("value", TokenType.ID),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""key = value;"""),
#             [
#                 ("key", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("value", TokenType.ID),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""number = 123.456;"""),
#             [
#                 ("number", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("123.456", TokenType.NUM),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""number = 123.;"""),
#             [
#                 ("number", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("123.", TokenType.NUM),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""number = -123.;"""),
#             [
#                 ("number", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("-123.", TokenType.NUM),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""number = 0xff;"""),
#             [
#                 ("number", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("0xff", TokenType.HEX),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""number = 0xffffffffffff;"""),
#             [
#                 ("number", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("0xffffffffffff", TokenType.HEX),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""number = -0x0023f9af12;"""),
#             [
#                 ("number", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("-0x0023f9af12", TokenType.HEX),
#                 (";", TokenType.SEMICOLON),
#             ]
#         ),
#         (
#             ("""
#              # This is a comment
#              key = value;
#              # This is another comment
#              key:value;
#              """),
#             [
#                 ("key", TokenType.ID),
#                 ("=", TokenType.EQUAL),
#                 ("value", TokenType.ID),
#                 (";", TokenType.SEMICOLON),
#                 ("key", TokenType.ID),
#                 (":", TokenType.COLON),
#                 ("value", TokenType.ID),
#                 (";", TokenType.SEMICOLON),
#             ]
#         )
#     ])
#     def test_parse_all_tokens(self, filecontents: str, tokenlst: list[tuple[str, TokenType]]):
#         memfile = io.StringIO(filecontents)
#
#         fstream = InputFileStream.FromFilePtr(memfile, 4)
#         tokstream = TokenStream(fstream)
#         index = 0
#         for token in tokstream:
#             self.assertLess(index, len(tokenlst))
#             expected_token = tokenlst[index]
#             self.assertEqual(str(token), expected_token[0])
#             self.assertEqual(token.type, expected_token[1]) # type: ignore
#             index += 1
#         self.assertEqual(index, len(tokenlst))
#
#     def test_eof_with_unidentified_token(self):
#         contents = """
#         "This is a string that is unterminated
#         """
#         memfile = io.StringIO(contents)
#
#         fstream = InputFileStream.FromFilePtr(memfile, 4)
#         tokstream = TokenStream(fstream)
#
#         token = tokstream.next()
#         self.assertIsNotNone(token)
#         self.assertEqual(token.type, TokenType.INVALID)
#         self.assertEqual(str(token), """        "This is a string that is unterminated""")
#
#         token = tokstream.next()
#         self.assertIsNone(token)

class TestToken(Enum):
    A = 0
    B = 1
    C = 2

class TestGrammar(Grammar):
    def get(self):
            return [
                [ TestToken.A, TestToken.B, TestToken.C ],
                [ TestToken.A, TestToken.C, TestToken.C ],
            ]

class GrammarTreeTest(unittest.TestCase):
    
    def test_grammar_tree_gen(self):
        gtree = GrammarTree.InitFromGrammar(TestGrammar())
        self.assertEqual(gtree.root.token, TokenGroupType.START)

        # START
        #   |
        #   A
        #   |  \
        #   |   \
        #   B    C
        #   |    |
        #   |    |
        #   C    C
        #   |    |
        #   |    |
        #  END  END


        # START -> A
        children = gtree.root.children
        self.assertEqual(1, len(children))
        
        # A -> [B, C]
        mid_children = children[0].children
        self.assertEqual(2, len(mid_children))

        # B -> C
        children = mid_children[0].children
        self.assertEqual(1, len(children))

        # C -> END
        children = children[0].children
        self.assertEqual(1, len(children))
        
        # END -> None
        children = children[0].children
        self.assertEqual(0, len(children))

        # C -> C
        children = mid_children[1].children
        self.assertEqual(1, len(children))
        
        # C -> END
        children = children[0].children
        self.assertEqual(1, len(children))

        # END -> None
        children = children[0].children
        self.assertEqual(0, len(children))

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

