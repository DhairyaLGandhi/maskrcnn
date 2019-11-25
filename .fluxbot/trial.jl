using GitHub
using GitHub.HTTP
using GitHub.JSON
using Sockets

const myauth = GitHub.authenticate(ENV["FLUXBOT_GITHUB_TOKEN"])
const trigger = r"FluxBot: .*"
const PROJECT = "15168210" # Flux.jl

const dic = Dict{String, Dict}("build" => Dict("body" => "Here are your results: "),
                                "feed" => Dict("body" => "It says not to!!"),)

const std_resp = Dict("body" => "I couldn't get that. Maybe try asking for `commands`?")
const failed_resp = Dict("body" => "Pipeline Failed: ")
placeholder_resp(issue, pl) = Dict("body" => "Alright, I'll respond here when I have results for pipeline for #$issue. The pipeline `$(pl["id"])` can be found at $(pl["web_url"])")

"""
checkout for key `pull_request` in event.payload["issue"]
"""
isPR(dict) = haskey(dict, "pull_request")

get_model_names(s) = join(split(s, ' ')[3:end], ' ')
get_command_name(s) = split(s, ' ')[2]

"""
  get_response(command, output = "")

Generate the params dict required to send a comment

Generally is fine as is, but the optional
output field can be appended to the message, for when
the `build` command needs to return the gist url

Potential security risk, in that we don't want it to
leak any data through this `output`.
"""
function get_response(command, output = "")
  if haskey(dic, command)
    val = copy(dic[command])
  else
    val = std_resp
  end
  if command == "commands"
    val["body"] = "Here are the commands: $(join(keys(dic), ' '))"
    val["body"] = val["body"] * " commands"
  end
  if command == "build"
    val["body"] = val["body"] * string(output)
  end
  val
end

"""
    no_existing_pipelines()

Checks that the current project does not have any pending or running pipelines.

Needed to ensure we get the correct artifacts out, since GitLab only serves the
most recent artifacts through its API.
"""
function no_existing_pipelines()
  d = Dict("status"=>"pending")
  pending = HTTP.get("https://gitlab.com/api/v4/projects/$PROJECT/pipelines", query = d).body |> String
  pending = pending == "[]"
  
  d = Dict("status"=>"running")
  running = HTTP.get("https://gitlab.com/api/v4/projects/$PROJECT/pipelines", query = d).body |> String
  running = running == "[]"

  running == pending == false
end

"""
    trigger_pipeline(id, model; ref = "master", token)
Triggers the model-zoo build with the given
PR against the specified models.

Returns the output of the curl call as a `Dict`.

NOTE: Checks for an existing running pipeline, so
the artifacts generated are consistent.
"""
function trigger_pipeline(id, model; ref = "master", token = ENV["GITLAB_MODELZOO_TOKEN"])
  # replace project with 15454378

  # Check existing running pipelines triggered by bot
  r = if no_existing_pipelines()
    read(`curl -X POST
       -F "token=$token"
       -F "ref=$ref"
       -F "variables[FLUXBOT]=true"
       -F "variables[PRID]=$id"
       -F "variables[TESTSUITE]=$model"
       -F "variables[MODELZOO_PIPELINE_ID]=98138293"
       https://gitlab.com/api/v4/projects/$PROJECT/trigger/pipeline`, String) |> JSON.parse
  end
end

function trial()
  @info ENV["GITHUB_EVENT_PATH"]
  f = JSON.parsefile(ENV["GITHUB_EVENT_PATH"])
  @show keys(f)
  event = GitHub.event_from_payload!("issue_comment", f)

  if event.payload["action"] == "deleted"
    return HTTP.Response(200)
  end

  # Ignore non-collaborators
  repo = event.repository
  user = event.payload["comment"]["user"]["login"]
  iscollab = GitHub.iscollaborator(repo, user; auth = myauth)
  if !iscollab
    return HTTP.Response(200)
  end

  if isPR(event.payload["issue"])
    phrase = event.payload["comment"]["body"]
    phrase = match(trigger, phrase)

    # return if the comment does not match regex
    if phrase == nothing
      return HTTP.Response(200)
    end

    # Common things
    com = get_command_name(phrase.match)
    model = get_model_names(phrase.match)
    comment_kind = :issue
    reply_to = event.payload["issue"]["number"]

    if com != "build"
      GitHub.create_comment(event.repository, reply_to, comment_kind;
                            auth = myauth,
                            params = get_response(com))
    else
      @assert com == "build"
      # resp = trigger_pipeline(reply_to, model)
      resp = Dict("id" => "98138293", "web_url" => "https://gitlab.com/JuliaGPU/Flux.jl/pipelines/98138293")
      if resp == nothing
        f_resp = copy(failed_resp)
        f_resp["body"] = f_resp["body"] * "There is an existing pipeline running. Abort that before triggering a new one."
        GitHub.create_comment(event.repository, reply_to, comment_kind,
                              auth = myauth,
                              params = f_resp)
        return HTTP.Response(200)
      end

      GitHub.create_comment(event.repository, reply_to, comment_kind,
                            auth = myauth,
                            params = placeholder_resp(reply_to, resp))
    end

    return HTTP.Response(200)
  end

  return HTTP.Response(500)
end

trial()
