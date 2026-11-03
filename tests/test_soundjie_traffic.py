import importlib
import unittest

class TrafficLabelTests(unittest.TestCase):
    def setUp(self):
        try: self.mod=importlib.import_module('scripts.host.prepare_soundjie_traffic_data')
        except ModuleNotFoundError: self.mod=None
        self.assertIsNotNone(self.mod,'traffic label preparation not implemented')

    def test_ambulance_requires_siren_and_unambiguous_source_title(self):
        self.assertEqual(self.mod.labels_for(['Siren'],'Ambulance siren.wav'),([1,0,0],[1,0,0]))
        self.assertIsNone(self.mod.labels_for(['Vehicle'],'Ambulance.wav'))
        self.assertIsNone(self.mod.labels_for(['Siren'],'police or ambulance'))

    def test_police_is_an_explicit_ambulance_negative(self):
        self.assertEqual(self.mod.labels_for(['Siren'],'Police siren'),([0,0,0],[1,0,0]))

    def test_traffic_can_be_positive_without_speech_claim(self):
        self.assertEqual(self.mod.labels_for(['Traffic_noise_and_roadway_noise'],'Road'),([0,1,0],[1,1,0]))

    def test_generic_siren_is_not_relabelled_as_ambulance(self):
        self.assertIsNone(self.mod.labels_for(['Siren'],'Warning sound'))

if __name__=='__main__': unittest.main()
