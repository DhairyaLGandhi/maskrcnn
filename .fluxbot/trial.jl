using GitHub
using GitHub.HTTP
using GitHub.JSON
using Sockets

const myauth = GitHub.authenticate(ENV["FLUXBOT_GITHUB_TOKEN"])
const trigger = r"FluxBot: .*"

const dic = Dict{String, Dict}("build" => Dict("body" => "Here are your results: "),
                                "feed" => Dict("body" => "It says not to!!"),)

const std_resp = Dict("body" => "I couldn't get that. Maybe try asking for `commands`?")
const failed_resp = Dict("body" => "Pipeline Failed: ")
placeholder_resp(issue, pl) = Dict("body" => "Alright, I'll respond here when I have results for pipeline for $issue.\n
                                              The pipeline $(pl["id"]) can be found at $(pl["web_url"])")

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

function trial()
  @info ENV["GITHUB_EVENT_PATH"]
  f = JSON.parsefile(ENV["GITHUB_EVENT_PATH"])
  @show keys(f)
  event = GitHub.event_from_payload!("issue_comment", f)

  if event.payload["action"] == "deleted"
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

      GitHub.create_comment(event.repository, reply_to, comment_kind,
                            auth = myauth,
                            params = placeholder_resp(reply_to, resp))
    end

    return HTTP.Response(200)
  end

  return HTTP.Response(500)
end

trial()
