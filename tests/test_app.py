import sys

sys.path.append("./python_service")
from app import api
from fastapi.testclient import TestClient

client = TestClient(api)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == '"Main page"'


def test_404():
    """
    Test is 404 is returned when a route is not found.
    """
    response = client.get("/user")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_right():
    response = client.get("/right")
    assert response.status_code == 200
    assert (
        response.text
        == '"something wise: yesterday is history, tomorrow is a mystery, but today is a gift. That is why it is called the present."'
    )  # noqa
