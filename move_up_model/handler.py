import json
# from move_ups import move_up_model

def move_up(event, context):
    try:
        with open('./example_output.json') as f:
            response = json.load(f)
        # self.response = move_up_model(event, "3244").output()
        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }
    except Exception:
        return {
            "statusCode": 500,
            "body": 'An error occured running model'
        }
