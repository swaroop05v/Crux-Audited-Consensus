import unittest

from crux.data.choices import ChoiceParseError, parse_choices


class ParseChoicesTest(unittest.TestCase):
    def test_ten_options_with_commas_in_text(self):
        raw = ("A. 46,000 g/mole, B. 5,030 g/mole, C. 50.3 kg/mole, D. 150,000 g/mole, E. 50,300 g/mole, "
               "F. 92,000 g/mole, G. Both 50,300 g/mole and 50.3 kg/mole are correct, H. 603 g/mole, "
               "I. 10 g/mole, J. 273 g/mole")
        choices = parse_choices(raw)
        self.assertEqual(list(choices), list("ABCDEFGHIJ"))
        self.assertEqual(choices["E"], "50,300 g/mole")
        self.assertEqual(choices["G"], "Both 50,300 g/mole and 50.3 kg/mole are correct")

    def test_space_before_comma_separator(self):
        raw = "A. Liquid biopsy (ctDNA) , B. Digital tomosynthesis , C. Elastography , D. Core needle biopsy"
        self.assertEqual(parse_choices(raw)["A"], "Liquid biopsy (ctDNA)")

    def test_newline_separated(self):
        raw = "A. Patient 2, Patient 1\nB. Patient 3, Patient 1\nC. Patient 1, Patient 2"
        self.assertEqual(parse_choices(raw), {"A": "Patient 2, Patient 1", "B": "Patient 3, Patient 1",
                                              "C": "Patient 1, Patient 2"})

    def test_label_like_text_inside_option_is_kept(self):
        choices = parse_choices("A. vitamin C. deficiency, B. vitamin D. excess")
        self.assertEqual(choices, {"A": "vitamin C. deficiency", "B": "vitamin D. excess"})

    def test_dict_passthrough(self):
        self.assertEqual(parse_choices({"A": " x ", "B": "y"}), {"A": "x", "B": "y"})

    def test_unparseable(self):
        with self.assertRaises(ChoiceParseError):
            parse_choices("no labels here")


if __name__ == "__main__":
    unittest.main()
