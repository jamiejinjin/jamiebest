import hashlib


def hash_the_password(password, salt):
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest()[:30]
