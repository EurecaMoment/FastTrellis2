"""Optional real OpenAI SDK transport test; model execution remains a CPU fake."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import test_openai_api as api_tests


@unittest.skipUnless(importlib.util.find_spec('openai'), 'Install openai for the optional SDK check')
class OpenAISDKTests(unittest.TestCase):
    def test_models_extensions_and_streaming_file_download(self):
        from openai import OpenAI
        fixture = api_tests.ApiTests()
        fixture.setUp()
        try:
            with OpenAI(base_url=fixture.url + '/v1', api_key='test-key', max_retries=0) as client:
                self.assertEqual(client.models.list().data[0].id, 'fasttrellis2')
                self.assertEqual(client.models.retrieve('fasttrellis2').id, 'fasttrellis2')
                job = client.post('/objects/generations', cast_to=dict,
                    body={'model': 'fasttrellis2', 'image': fixture.image, 'wait': True})
                self.assertEqual(job['status'], 'completed')
                self.assertEqual(client.get('/objects/generations/' + job['id'], cast_to=dict)['id'], job['id'])
                file_id = job['data'][0]['file_id']
                self.assertEqual(client.files.retrieve(file_id).id, file_id)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'object.glb'
                    with client.files.with_streaming_response.content(file_id) as response:
                        response.stream_to_file(path)
                    self.assertEqual(path.read_bytes()[:4], b'glTF')
                spec = client.get('/../openapi.json', cast_to=dict)
                self.assertEqual(spec['openapi'], '3.1.0')
                if importlib.util.find_spec('jsonschema'):
                    from jsonschema import Draft202012Validator
                    for schema in spec['components']['schemas'].values():
                        Draft202012Validator.check_schema(schema)
                    schema = {**spec['components']['schemas']['Generation'], 'components': spec['components']}
                    Draft202012Validator(schema).validate(job)
                self.assertEqual(client.delete('/objects/generations/' + job['id'], cast_to=dict)['status'], 'completed')
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
