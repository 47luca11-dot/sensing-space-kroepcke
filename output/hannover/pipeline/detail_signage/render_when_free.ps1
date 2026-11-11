# waits until >= 5 GB RAM are free, then renders the given preview views with headless Blender
param([string]$Tag = "-", [string]$Views = "rathenaustr_nah karmarschstr_nah", [int]$MaxMin = 40)
$blender = "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
$script = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/pipeline/detail_signage/preview_blender.py"
$sign = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/detail_raw/signage.glb"
$t0 = Get-Date
while ($true) {
  $f = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
  if ($f -ge 5) { break }
  if (((Get-Date) - $t0).TotalMinutes -gt $MaxMin) { "TIMEOUT free={0:N2}" -f $f; exit 1 }
  Start-Sleep -Seconds 20
}
"START free={0:N2} GB after {1:N1} min" -f $f, ((Get-Date) - $t0).TotalMinutes
$args2 = @("-b", "--factory-startup", "-P", $script, "--", $Tag, $sign) + ($Views -split " ")
& $blender @args2 2>&1 | Select-String -Pattern "RENDERED|Error|error|Traceback" | ForEach-Object { $_.Line }
"DONE"
