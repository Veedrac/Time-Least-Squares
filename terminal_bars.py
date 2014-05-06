node      = "█"
remainder = " ▏▎▍▌▋▊▉"
overflowstr = "▓▓▒▒░░"
background = " "

barparts = {
	"topleft":  "┏",
	"top":      "━",
	"topright": "┓",

	"left": "┃",
	"right": "┃",

	"bottomleft": "┗",
	"bottom":      "━",
	"bottomright": "┛",
}

def plot(names, data, width, *, formatter="{}".format, maximum=None):
	namespace = max(map(len, names)) + 3
	
	formatted_data = [formatter(datum) for datum in data]
	dataspace = max(map(len, formatted_data))

	barspace = width-4 - namespace - dataspace
	overflow = overflowstr.rjust(barspace, node)

	bartop = "{topleft}{top}{topright}".format(
		topleft  = barparts["topleft"],
		top      = barparts["top"] * barspace,
		topright = barparts["topright"],
	)

	barbottom = "{bottomleft}{bottom}{bottomright}".format(
		bottomleft  = barparts["bottomleft"],
		bottom      = barparts["bottom"] * barspace,
		bottomright = barparts["bottomright"],
	)

	print(" "*namespace, bartop)

	for name, datum, datumstr in zip(names, data, formatted_data):
		if maximum is not None:
			datum = datum * barspace / maximum
		else:
			datum /= len(remainder)

		if datum > barspace:
			bar = overflow
		else:
			notches = round(datum * len(remainder))

			full, partial = divmod(notches, 8)
			bar = node * full + remainder[partial].rstrip()

		print("{name:>{namespace}} {left}{bar:{background}<{barspace}}{right} {datum}".format(
			name=name,
			namespace=namespace,
			left=barparts["left"],
			bar=bar,
			background=background,
			barspace=barspace,
			right=barparts["right"],
			datum=datumstr
		))

	print(" "*namespace, barbottom)
