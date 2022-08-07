import os

import redis
from rq import Connection, Queue, Worker

listen = ["default"]

redis_conn = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis-15346.c240.us-east-1-3.ec2.cloud.redislabs.com"),
    port=os.getenv("REDIS_PORT", "15346"),
    password=os.getenv("REDIS_PASSWORD", "46cc0AdOOBg0vKnk7JfOJV0rH16Zz399"),
)


if __name__ == "__main__":
    with Connection(redis_conn):
        print("connection")
        worker = Worker(list(map(Queue, listen)))
        worker.work()
