
import os
import boto.s3.connection

class Env:
    __evn_dist = os.environ

    @staticmethod
    def judge_env(env, message):
        env_value = Env.__evn_dist.get(env)
        if env_value is None:
            raise RuntimeError(message + env)

        return env_value


class ConnectS3(object):
    def __init__(self, bucket, model_key, network_key=None):
        self.conn = boto.connect_s3(
            aws_access_key_id = Env.judge_env("S3_ACCESS_KEY", "s3_access_key"),
            aws_secret_access_key = Env.judge_env("S3_SECRET_KEY", "s3_secret_key"),
            host = Env.judge_env("S3_HOST", "s3_host"),
            port = int(Env.judge_env("S3_PORT", "s3_port")),
            is_secure=False,  # uncomment if you are not using ssl
            calling_format = boto.s3.connection.OrdinaryCallingFormat(),
        )

        self.model_key = model_key
        self.create_bucket(bucket)

        model_type = Env.judge_env("MODEL_TYPE", "model_type")
        if model_type == "PT":
            self.network_key = network_key
            self.set_pt()
        self.set_model_data()

    def create_bucket(self, bucket):
        self.bucket = self.conn.create_bucket(bucket)

    def set_model_data(self):
        key = self.bucket.get_key(self.model_key)
        key.get_contents_to_filename("./model.m")

    def set_pt(self):
        key = self.bucket.get_key(self.network_key)
        key.get_contents_to_filename("./Network.py")

#s3 = ConnectS3('kubeflow-anonymous-test', "test/2019-12-16T15-47-38Z/test", 'test/2019-12-16T17-43-41Z/network')