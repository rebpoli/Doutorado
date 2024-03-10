
pdf:
	mkdir -p _temp && rsync -am * _temp/ 
	cd _temp && nbtopdf CP5
	mv _temp/CP5.pdf .

clean:
	rm -rf _temp
