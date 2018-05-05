read_file = open("craft-2.0/articles/txt/12079497.txt","r")
out_file = open("line_out.txt","w")
ind_lines = []
for line in read_file:
	line = line.strip()
	#Non-empty line
	if line != "":
		ind_lines += line.split(". ")
#Write the lines to output file
for ind_line in ind_lines:
	out_file.write(ind_line + "\n")
out_file.close()
read_file.close()

