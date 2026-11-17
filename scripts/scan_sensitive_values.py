#!/usr/bin/env python3
"""Scan Git objects without printing matched content."""
import argparse
import io
import json
import re
import subprocess
import sys
from pathlib import Path


def git(*args, input=None):
    return subprocess.run(['git', *args], input=input, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def entries(tree):
    raw = git('ls-files', '-s', '-z') if tree is None else git('ls-tree', '-r', '-z', tree)
    for item in raw.split(b'\0'):
        if not item:
            continue
        meta, path = item.split(b'\t', 1)
        fields = meta.split()
        if tree is None:
            if fields[2] != b'0':
                raise ValueError('Unmerged index entry')
            mode, oid = fields[0], fields[1]
        else:
            mode, kind, oid = fields
            if kind != b'blob':
                continue
        if mode == b'160000':
            continue
        yield path.decode('utf-8', 'surrogateescape'), oid.decode('ascii')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['staged', 'tracked', 'outgoing'])
    args = parser.parse_args()
    root = Path(git('rev-parse', '--show-toplevel').decode().strip())
    local = root / 'account-parameters.json'
    identifiers = []
    if local.is_file():
        cfg = json.loads(local.read_text(encoding='utf-8-sig'))
        identifiers = [str(cfg['aws']['account_id']), str(cfg['cognito']['user_pool_id'])]
        if not all(identifiers):
            raise ValueError('Missing local identifier')
    patterns = [
        ('AWS access key', re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
        ('private key', re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----')),
        ('account-specific ARN', re.compile(rb'arn:aws(?:-us-gov|-cn)?:iam::(?!000000000000)[0-9]{12}:(?:role/|oidc-provider/)')),
    ]
    trees = [None]
    if args.mode == 'outgoing':
        trees = []
        updates = 0
        for line in sys.stdin:
            updates += 1
            parts = line.split()
            if len(parts) != 4:
                raise ValueError('Malformed pre-push input')
            _, local_sha, _, remote_sha = parts
            if set(local_sha) == {'0'}:
                continue
            if not re.fullmatch(r'[0-9a-fA-F]{40,64}', local_sha + '' ):
                raise ValueError('Invalid revision')
            cmd = ['rev-list', local_sha]
            if set(remote_sha) != {'0'}:
                cmd.extend(['--not', remote_sha])
            trees.extend(git(*cmd).decode().splitlines())
        if not updates:
            raise ValueError('Missing pre-push input')
        trees = list(dict.fromkeys(trees))
    seen = set()
    failures = set()
    for tree in trees:
        seen.update(entries(tree))
    objects = sorted({oid for _, oid in seen})
    stream = io.BytesIO(git('cat-file', '--batch', input=('\n'.join(objects) + '\n').encode()))
    blobs = {}
    for oid in objects:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError('Unable to read Git blob')
        size = int(header[2])
        blobs[oid] = stream.read(size)
        if stream.read(1) != b'\n':
            raise ValueError('Invalid Git blob output')
    for path, oid in sorted(seen):
            lower = path.lower().replace('\\', '/')
            if lower == 'account-parameters.json' or lower.startswith('.generated/') or lower.endswith(('.pem', '.key', '.p12', '.pfx')):
                failures.add((path, 'prohibited file'))
            content = blobs[oid]
            for label, pattern in patterns:
                if pattern.search(content):
                    failures.add((path, label))
            if 'data' not in lower.split('/'):
                for value in identifiers:
                    if value.encode() in content:
                        failures.add((path, 'local deployment identifier'))
    for path, label in sorted(failures):
        print(f'{path}: {label}', file=sys.stderr)
    if failures:
        return 1
    print('Security scan completed; data content was checked for credential patterns only.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Security scan could not complete: {type(exc).__name__}', file=sys.stderr)
        sys.exit(2)
