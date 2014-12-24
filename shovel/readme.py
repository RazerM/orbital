from shovel import task
import subprocess


@task
def rst():
    """Convert markdown readme to reStructuredText"""
    subprocess.call(['pandoc', '--from=markdown', '--to=rst', '--output=README', 'README.md'])
