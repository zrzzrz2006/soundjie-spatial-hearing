import unittest

from scripts.host.assess_soundjie_siren5_board import decide, merge, parse_output


class PhysicalProbeProtocol(unittest.TestCase):
    def test_merge_changes_winner_after_softmax(self):
        top, ms, scores = parse_output(b"2 .35 1.25 .25 .25 .35 .05 .05 .05\n")
        self.assertEqual(top, 2)
        self.assertEqual(ms, 1.25)
        self.assertEqual(decide(merge(scores)), 0)

    def test_bad_vector_or_metadata_is_not_physical_evidence(self):
        for line in (b"0 .4 1 .4 .1 .1 .1 .1 .1",
                     b"0 .5 1 .4 .1 .1 .1 .1 .2",
                     b"2 .4 1 .4 .1 .1 .1 .1 .2",
                     b"0 .4 1 .4 .1 .1 nan .1 .2",
                     b"0 .4 1 .4 .1 .1 .1 .3"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_output(line)

    def test_threshold_rejection_is_not_argmax_agreement(self):
        _, _, scores = parse_output(b"2 .6 1 .05 .05 .6 .1 .1 .1")
        self.assertEqual(decide(merge(scores)), 4)


if __name__ == "__main__":
    unittest.main()
