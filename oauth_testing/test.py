from authomatic import Authomatic
from authomatic.providers import oauth2

CONFIG = {
    'github': {
        'class_': oauth2.GitHub,
        'consumer_key': 'dab0e3b0d13510195957',
        'consumer_secret': '3da0cd40ffe175bfba078f4398fe6a1c5fce5c7b',
        'scope': oauth2.GitHub.user_info_scope,
        '_apis': {
            'Get your events': ('GET', 'https://api.github.com/users/{user.username}/events'),
            'Get your watched repos': ('GET', 'https://api.github.com/user/subscriptions'),
        },
    },
}

authomatic = Authomatic(CONFIG, 'secret')