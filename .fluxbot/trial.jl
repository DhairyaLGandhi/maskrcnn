using GitHub
using GitHub.HTTP
using GitHub.JSON
using Sockets

const myauth = GitHub.authenticate(ENV["FLUXBOT_GITHUB_TOKEN"])

function trial()
  @info ENV["GITHUB_EVENT_PATH"]
  f = JSON.parsefile(ENV["GITHUB_EVENT_PATH"])
  @show keys(f)
  event = GitHub.event_from_payload!("issue_comment", f)
  @show event

  if event.payload["action"] == "deleted"
    return HTTP.Response(200)
  end

  @show event.payload["comment"]["body"]
  # comment_kind = :issue
  # reply_to = event.payload["issue"]["number"]
  # GitHub.create_comment(event.repository, reply_to, comment_kind;
  #                       auth = myauth,
  #                       params = Dict("body" => "Have a reply!"))
end

trial()
