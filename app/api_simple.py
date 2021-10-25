import hug

@hug.get("/")
def api_simple():
    return "hello"