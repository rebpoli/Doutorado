
pdf:
	mkdir -p _temp && rsync -am * _temp/ 
	cd _temp && nbtopdf CP8
	mv _temp/CP8.pdf .

clean:
	rm -rf _temp
