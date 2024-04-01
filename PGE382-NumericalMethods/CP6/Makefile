
pdf:
	mkdir -p _temp && rsync -am * _temp/ 
	cd _temp && nbtopdf CP6
	mv _temp/CP6.pdf .

clean:
	rm -rf _temp
