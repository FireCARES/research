import json
from move_ups import move_up_model

def move_up(event, context):
    print(event);
    try:
        model = move_up_model(event)
        return {
            "statusCode": 200,
            "body": json.dumps(model.output)
        }
    except Exception as error:
        print(error)
        return {
            "statusCode": 500
        }

