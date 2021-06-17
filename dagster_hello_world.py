from dagster import pipeline, solid, execute_pipeline


@solid
def get_name():
    return "dagster"


@solid
def hello(context, name: str):
    context.log.info(f"Hello, {name}!")


@pipeline
def hello_pipeline():
    hello(get_name())



if __name__ == "__main__":
    result = execute_pipeline(hello_pipeline)
