"""Explicit product rule bindings; native v0 game semantics remain frozen."""
import json
from . import config

CONTRACT = dict(schema='congklak-product-rules-v1', profile=config.PROFILE,
                rules_digest=config.RULES_DIGEST, opening='sequential',
                rounds='single', pits_per_side=7, initial_seeds_per_pit=7)
SOURCE = 'https://kebudayaan.kemdikbud.go.id/ditwdb/congklak-betawi/'


def validate(value):
    if not isinstance(value, dict) or config.canonical(value) != config.canonical(CONTRACT):
        raise config.InputError('rule_profile_mismatch',
            'This table uses the sequential, single-round Betawi profile. The imported rule profile differs; saved games cannot be reinterpreted under another opening.')


def engine_identity(identity):
    if not isinstance(identity, dict) or identity.get('profile') != config.PROFILE or identity.get('rules_digest') != config.RULES_DIGEST:
        raise config.InputError('rule_profile_mismatch', 'Saved analysis uses different game rules.')


def require(db, record_id):
    row = db.execute('SELECT body,sha FROM product_rules WHERE id=?', (record_id,)).fetchone()
    if row is None:
        raise config.InputError('missing_rule_binding', 'Saved game has no rule binding. Restart to migrate legacy games safely.')
    value = json.loads(row['body'])
    if config.digest(value) != row['sha'] or set(value) != {'contract','origin'} or value['origin'] not in ('legacy_migration','new_save'):
        raise config.InputError('corrupt_rule_binding', 'Saved rule binding checksum or schema differs.')
    validate(value['contract'])
    return value


def bind(db, record_id, origin):
    # Never replace an existing record: a foreign or damaged identity must fail closed.
    if db.execute('SELECT 1 FROM product_rules WHERE id=?', (record_id,)).fetchone():
        return require(db, record_id)
    body = dict(contract=dict(CONTRACT), origin=origin)
    db.execute('INSERT INTO product_rules VALUES(?,?,?)', (record_id,config.canonical(body),config.digest(body)))
    return body


def migrate(db):
    db.execute('CREATE TABLE IF NOT EXISTS product_rules(id TEXT PRIMARY KEY, body TEXT NOT NULL, sha TEXT NOT NULL)')
    with db:
        # All pre-binding product releases used this exact frozen v0 profile.
        bind(db, 'current', 'legacy_migration')
        for row in db.execute('SELECT id FROM sessions').fetchall():
            bind(db, row['id'], 'legacy_migration')
