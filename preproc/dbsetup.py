import json
import csv
import bz2
from pymongo import MongoClient
from dateutil import parser, relativedelta


class Importer:
    def __init__(self, db_name):
        client = MongoClient()
        self.db = getattr(client, db_name)

    def clear(self, col_name):
        collection = getattr(self.db, col_name)
        collection.delete_many({})

    def import_tweets_from_file(self, col_name, filename, json_type=False, borked_json=False):
        """
        Read tweet objects from single file and insert into db collection
        :param col_name: Collection name to insert records into
        :param filename: File path to read from
        :param json_type: File is in JSON format
        :param borked_json: File is multi-line JSON objects without list separators
        """
        collection = getattr(self.db, col_name)
        with open(filename, encoding="utf8", mode='r') as infile:
            if json_type:
                if borked_json:
                    try:
                        content = infile.read().replace('}{', '}abz!!!$|,{')
                        content_list = content.split('abz!!!$|,')
                        for str_obj in content_list:
                            obj = json.loads(str_obj, strict=False)
                            if obj['lang'] in ['en', 'uk']:
                                collection.insert_one(obj)
                    except Exception as e:
                        print(e)
                else:
                    for line in infile:
                        try:
                            obj = json.loads(line, strict=False)
                            collection.insert_one(obj)
                        except Exception as e:
                            print(e)
            else:
                reader = csv.DictReader(infile)
                for row in reader:
                    if row['tweet_language'].lower() in ['en', 'uk']:
                        collection.insert_one(row)

    @staticmethod
    def get_int_for_float(str_val):
        try:
            return int(float(str_val))
        except ValueError:
            return 0

    @staticmethod
    def get_str_for_date(date_val):
        dt = parser.parse(date_val, fuzzy=True)
        dt = dt + relativedelta.relativedelta(year=2020)
        return dt

    @staticmethod
    def get_int(str_val):
        try:
            return int(str_val)
        except ValueError:
            return 0

    @staticmethod
    def create_entity_list(array, column, field):
        val_dict = array[column]
        values = [value[field] for value in val_dict]
        return "[" + ", ".join(values) + "]"

    def merge_tweets(self, merge_col, col_1, col_2):
        """
        Merge two collections of tweets to a third collection
        :param merge_col: Collection name to copy merged records to
        :param col_1: First collection to merge (must be astroturf-type)
        :param col_2: Second collection to merge (must be clean-type)
        :return: None
        """
        collection = getattr(self.db, merge_col)
        collection.delete_many({})

        reg_fields = [('full_text', 'tweet_text'),
                      ('hashtags', 'hashtags'),
                      ('urls', 'urls'),
                      ('user_description', 'user_profile_description'),
                      ('user_location', 'user_reported_location'),
                      ('user_name', 'user_display_name'),
                      ('user_screen_name', 'user_screen_name')]

        col1_posts = getattr(self.db, col_1).find({}, no_cursor_timeout=True)

        for post in col1_posts:
            obj = {}

            for new_field, date_field in [('created_at','tweet_time'), ('user_created_at', 'account_creation_date')]:
                obj[new_field] = self.get_str_for_date(post[date_field])

            for new_field, float_field in [('favorite_count', 'like_count'), ('retweet_count', 'retweet_count')]:
                obj[new_field] = self.get_int_for_float(post[float_field])

            for new_field, int_field in [('user_followers_count', 'follower_count'),
                                         ('user_following_count', 'following_count')]:
                obj[new_field] = self.get_int(post[int_field])

            for new_field, reg_field in reg_fields:
                obj[new_field] = post[reg_field]

            obj['user_profile_urls'] = post['user_profile_url'] if post['user_profile_url'] else ''

            obj['label'] = 'astroturf'
            collection.insert_one(obj)
        col1_posts.close()

        col2_posts = getattr(self.db, col_2).find({}, no_cursor_timeout=True)
        for post in col2_posts:
            obj = {'created_at': self.get_str_for_date(post['created_at']),
                   'user_created_at': self.get_str_for_date(post['user']['created_at'])}

            for float_field in ['favorite_count', 'retweet_count']:
                obj[float_field] = self.get_int_for_float(post[float_field])

            for new_field, nest_field, int_field in [('user_followers_count', 'user', 'followers_count'),
                                                     ('user_following_count', 'user', 'friends_count')]:
                obj[new_field] = self.get_int(post[nest_field][int_field])

            ent_array = post['entities']
            obj['hashtags'] = self.create_entity_list(ent_array, 'hashtags', 'text')
            obj['urls'] = self.create_entity_list(ent_array, 'urls', 'expanded_url')

            if 'full_text' in post.keys():
                obj['full_text'] = post['full_text']
            elif 'text' in post.keys():
                obj['full_text'] = post['text']
            else:
                raise KeyError("No key found in object for tweet text field.")
            obj['user_description'] = post['user']['description']
            obj['user_profile_urls'] = post['user']['url'] if post['user']['url'] else ''
            obj['user_location'] = post['user']['location']
            obj['user_name'] = post['user']['name']
            obj['user_screen_name'] = post['user']['screen_name']

            obj['label'] = 'none'
            collection.insert_one(obj)
        col2_posts.close()
