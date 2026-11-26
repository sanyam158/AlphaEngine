' Runs run_eod.cmd with no visible window (used by Windows Task Scheduler).
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
Set sh = CreateObject("WScript.Shell")
WScript.Quit sh.Run("""" & here & "\run_eod.cmd""", 0, True)
