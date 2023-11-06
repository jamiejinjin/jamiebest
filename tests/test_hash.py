import sys

sys.path.append(".")
from hashing import hash_the_password


def test_hash():
    password1 = "password"

    salt = "&^%$#"
    salt2 = "&^%$#1"

    assert hash_the_password(password1, salt) == hash_the_password(password1, salt)
    assert hash_the_password(password1, salt) != hash_the_password(password1, salt2)
