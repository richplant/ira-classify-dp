from pymongo import MongoClient
import pandas as pd


class Framer:
    def __init__(self, db_name):
        client = MongoClient()
        self.db = getattr(client, db_name)

    @staticmethod
    def get_from_db(collection):
        cursor = collection.find({})
        df = pd.DataFrame(list(cursor)).fillna(0)
        del df['_id']
        return df

    @staticmethod
    def frame_set(frame):
        frame['favorite_count'] = pd.to_numeric(frame['favorite_count'])
        frame['retweet_count'] = pd.to_numeric(frame['retweet_count'])
        frame['user_followers_count'] = pd.to_numeric(frame['user_followers_count'])
        frame['user_following_count'] = pd.to_numeric(frame['user_following_count'])
        frame['created_at'] = pd.to_numeric(frame['created_at'])
        frame['user_created_at'] = pd.to_numeric(frame['user_created_at'])
        frame['hashtags'] = frame['hashtags'].astype(str)
        frame['urls'] = frame['urls'].astype(str)
        frame['user_profile_urls'] = frame['user_profile_urls'].astype(str)
        frame['user_location'] = frame['user_location'].astype(str)
        frame['user_name'] = frame['user_name'].astype(str)
        frame['user_screen_name'] = frame['user_screen_name'].astype(str)
        frame['user_description'] = frame['user_description'].astype(str)
        frame['full_text'] = frame['full_text'].astype(str)
        return frame

    def get_frame(self, col_name):
        df = self.get_from_db(getattr(self.db, col_name))
        df = self.frame_set(df)
        return df

