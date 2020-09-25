# To compile:

Add the following make flags:

```
USE_TRANSMOGRIFIER=TRUE
USE_PERMISSIVE_SPACEDIM=TRUE
```

# Running

I was using the following inputs files to test this:

- inputs.noboost.2d.small
- inputs.noboost.3d.small

With the transmogrifier main, those have to be supplied in this order:

`./Castro3d.gnu.MPI.ex inputs.noboost.3d.small inputs.noboost.2d.small`
