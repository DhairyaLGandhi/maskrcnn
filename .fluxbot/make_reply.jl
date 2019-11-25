# This is executed inside GitLab, so all the variables from there should be available here.

using GitHub
using GitHub.HTTP, GitHub.JSON
using Glob

const myauth = GitHub.authenticate(ENV["BOT_SECRET"])


# Just link to where the arifacts for a job can be browsed
# Download artifacts
# Unzip and make into a gist
# Send response

artifacts_name = "artifacts_$CI_JOB_ID.zip"
download("https://gitlab.com/JuliaGPU/Flux.jl/-/jobs/artifacts/master/download?job=$CI_JOB_NAME", artifacts_name)
run(`unzip $artifacts_name`)
files = glob("*.ipynb", "notebooks")

gist_params = Dict("description" => "Build Results",
                    "public" => true,
                    "files" => Dict(f => Dict("content" => read(f,String),) for f in files))

g = GitHub.create_gist(auth = myauth, params = gist_params)

dict = Dict("body" => "Find the artifacts for `$CI_JOB_NAME` at $(g.html_url)")
GitHub.create_commment(GitHub.Repo(ENV["REPO_NAME"]), ENV["PRID"], :issue, auth = myauth, params = dict)