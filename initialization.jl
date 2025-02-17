import Pkg
Pkg.activate("./")
Pkg.instantiate()  # Ensure all dependencies listed in Project.toml are installed
using CondaPkg
CondaPkg.add("catboost"; version="=1.2.7")
Pkg.resolve()      # Ensure the package dependency graph is consistent
Pkg.update()       # Fetch the latest versions of your packages