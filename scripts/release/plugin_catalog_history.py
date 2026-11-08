"""Keep every published plugin release available to compatible engine versions."""
import copy
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

CATALOG_URL = 'https://downloads.infernux-engine.com/plugins/official-registry.json'
SCHEMA = 'infernux.official_plugin_registry'


def published_catalog():
    request = Request(CATALOG_URL, headers={'User-Agent': 'Infernux-Release-Publisher'})
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        return {'$schema': SCHEMA, 'packages': []}


def merge_history(published, incoming):
    """Append releases; an existing version cannot acquire another payload identity."""
    releases = {}
    for document in (published, incoming):
        if document['$schema'] != SCHEMA:
            raise ValueError('Unsupported plugin catalog schema')
        for item in document['packages']:
            key = (item['reference'].casefold(), item['version'])
            old = releases.get(key)
            if old is not None:
                for field in ('engine', 'artifact', 'source', 'repository'):
                    if old.get(field) != item.get(field):
                        raise ValueError(f'Published plugin identity is immutable: {key}: {field}')
            releases[key] = copy.deepcopy(item)
    return {'$schema': SCHEMA, 'packages': [releases[key] for key in sorted(releases)]}
