from cloup import group, help_option, version_option
from .. import __version__


@group('Ginjinn2 pollentube workflow')
@version_option(__version__, '-v', '--version')
@help_option('-h', '--help')
def main():
    pass


from .convert import convert
from .measure import measure

main.add_command(convert)
main.add_command(measure)
