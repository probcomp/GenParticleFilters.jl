using Documenter, GenParticleFilters

makedocs(
   sitename="GenParticleFilters.jl",
   format=Documenter.HTML(
      prettyurls=get(ENV, "CI", nothing) == "true",
      canonical="https://probcomp.github.io/GenParticleFilters.jl/stable",
      description="Documentation for GenParticleFilters.jl, " *
                  "a library for particle filtering and " *
                  "sequential Monte Carlo algorithms using Gen.",
   ),
   pages=[
      "Home" => "index.md",
      "Reference" => [
         "initialize.md",
         "update.md",
         "resample.md",
         "rejuvenate.md",
         "resize.md",
         "translate.md",
         "utils.md",
      ]
   ]
)

deploydocs(
    repo = "github.com/probcomp/GenParticleFilters.jl.git",
)
