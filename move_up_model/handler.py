import json
from move_ups import move_up_model

def move_up(event, context):
    try:
        print(event)
        model = move_up_model(event.body)
        return {
            "statusCode": 200,
            "body": json.dumps(model.output)
        }
    except Exception as error:
        print(error)
        return {
            "statusCode": 500
        }

