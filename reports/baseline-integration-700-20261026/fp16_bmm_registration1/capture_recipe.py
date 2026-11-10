import json,os,pathlib
pathlib.Path(__file__).with_name('RECIPE_ENV.json').write_text(json.dumps(dict(os.environ),indent=2))
