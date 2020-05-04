import json
from move_ups import move_up_model

def move_up(event, context):
    try:
        payload = json.loads(event['body'])
        model = move_up_model(payload)
        return {
            "statusCode": 200,
            "headers": {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': 'true',
            },
            "body": json.dumps(model.output)
        }
    except Exception as error:
        print(error)
        return {
            "statusCode": 500
        }

