# Detects if macOS
isMac()
{
    [ "$(uname)" == "Darwin" ]
}

# Detects if Linux OS
isLinux()
{
    [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]
}

# Detects if Ubuntu OS
isUbuntu()
{
    [ "$(expr substr $(awk -F= '/^NAME/{print $2}' /etc/os-release) 2 6)" == "Ubuntu" ]
}

# Detects if Windows OS
isWin()
{
    [ "$(expr substr $(uname) 1 5)" == "MINGW" ]
}

# Detects if arm architecture
isArm()
{
    [ "$(arch)" == "aarch64" ]
}
