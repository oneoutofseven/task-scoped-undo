"""Check that the optional structural constraints actually reach the local server."""
import io
import json
import unittest
from unittest.mock import patch
from task_undo.agents.model import ChatModel
from test_agent import CONFIG


class ResponseSchemaTests(unittest.TestCase):
    def test_schema_sent_and_default_preserved(self):
        model = ChatModel(CONFIG)
        schema = {'type':'object','properties':{'x':{'type':'string'}},'required':['x']}
        for supplied in [None, schema]:
            reply = {'choices':[{'finish_reason':'stop','message':{'content':'{"x":"a"}'}}]}
            with patch.object(model.opener, 'open', return_value=io.BytesIO(json.dumps(reply).encode())) as send:
                model.complete([{'role':'user','content':'test'}], response_schema=supplied)
            payload = json.loads(send.call_args.args[0].data)
            if supplied is None:
                self.assertEqual(payload['response_format'], {'type':'json_object'})
            else:
                self.assertEqual(payload['response_format']['type'], 'json_schema')
                self.assertTrue(payload['response_format']['json_schema']['strict'])
                self.assertEqual(payload['response_format']['json_schema']['schema'], schema)
            self.assertEqual(payload['temperature'], 0)
