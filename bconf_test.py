import io
import unittest
from bconf_lib import InputFileStream, TokenStream
from parameterized import parameterized

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
    def test_parse_tokens(self, filecontents: str, tokenlst: list[str], chunksize: int):
        memfile = io.StringIO(filecontents)

        fstream = InputFileStream.FromFilePtr(memfile, chunksize)
        tokstream = TokenStream(fstream)
        index = 0
        for token in tokstream:
            self.assertTrue(index < len(tokenlst))
            expected_token = tokenlst[index]
            self.assertEqual(str(token), expected_token)
            index += 1
        self.assertTrue(index == len(tokenlst))

if __name__ == '__main__':
    unittest.main()

