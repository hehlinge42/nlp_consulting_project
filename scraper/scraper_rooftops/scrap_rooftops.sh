#!/bin/bash

#array=("https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d15043073-Reviews-Savage_Garden-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d10500858-Reviews-Forza_Win-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d12900577-Reviews-Pergola_on_the_Roof-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d11908107-Reviews-Aviary-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d8366506-Reviews-Radio_Rooftop-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d16385768-Reviews-FEST_Camden-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d11638827-Reviews-Boundary_Rooftop-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d18850403-Reviews-Seabird-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d1391383-Reviews-Queen_of_Hoxton-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d6893599-Reviews-The_Culpeper-London_England.html"
#		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d4295544-Reviews-SUSHISAMBA_Heron_Tower-London_England.html"
#		"https://www.tripadvisor.fr/Restaurant_Review-g186338-d2179907-Reviews-The_Rooftop-London_England.html")

array=("https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d15348913-Reviews-Art_Yard_Bar_Kitchen-London_England.html"
		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d5220272-Reviews-Golden_Bee-London_England.html"
		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d18719167-Reviews-Allegra-London_England.html"
		"https://www.tripadvisor.co.uk/Restaurant_Review-g186338-d19425651-Reviews-The_Nest_in_Treehouse-London_England.html")

i=-14
for url in "${array[@]}"; do   # The quotes are necessary here
	echo LAUNCHING SPIDER
    scrapy crawl BokanSpider -a debug=1 -a root_url=$url -a id_resto=$i
	i=$((i-1))
done
