from dotenv import load_dotenv
from tui import WikiSearchApp

load_dotenv()


def run() -> None:
    WikiSearchApp().run()


if __name__ == "__main__":
    run()
