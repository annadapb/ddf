import click

@click.command()
@click.option('-x', 'x value')
@click.option('-y', 'x value')
@click.option('-z', 'x value')
def main(x, y, z):
    print(x, y, z)

if __name__=='__main__':
    main()
    pass
