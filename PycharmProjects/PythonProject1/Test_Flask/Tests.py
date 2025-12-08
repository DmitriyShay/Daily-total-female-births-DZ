import unittest
import sys
import os

from pandas._libs import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import app as tested_app
import join

class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        tested_app.app.config['TESTING'] = True
        self.app = tested_app.app.test_client()

    def test_get_hello_endpoint(self):
        r = self.app.get('/')
        self.assertEqual(r.data, b'Hello, World!')

    def test_post_hello_endpoint(self):
        r = self.app.post('/')
        self.assertEqual(r.status_code, 405)

    def test_get_api_endpoint(self):
        r = self.app.get('/api')
        self.assertEqual(r.json, {'status': 'test'})

    def test_correct_post_api_endpoint(self):
        r = self.app.post('/api',
                          content_type='application/json',
                          data=json.dumps({'name': 'Den', 'age': 100}))
        self.assertEqual(r.json, {'status': 'ok'})
        self.assertEqual(r.status_code, 200)

    def test_add_succcess(self):
        r = self.app.get('/add?a=3&b=2')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data, b'6.0')