// Package embedded holds files compiled into the agentic-search binary.
package embedded

import _ "embed"

//go:embed SKILL.md
var SkillMD string
