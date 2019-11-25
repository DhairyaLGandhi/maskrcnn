using GitHub
using GitHub.HTTP
using GitHub.JSON
using Sockets

const myauth = GitHub.authenticate(ENV["FLUXBOT_GITHUB_TOKEN"])

function trial()
  @info ENV["GITHUB_EVENT_PATH"]

end

trial()
