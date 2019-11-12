mkdir data
mkdir data/convex
mkdir data/kb

wget http://qa.mpi-inf.mpg.de/convex/data.zip
unzip data.zip -d $(pwd)/data/convex
mv $(pwd)/data/convex/data $(pwd)/data/convex/cache
mv $(pwd)/data/convex/cache/stopwords.txt $(pwd)/data/convex/
mv $(pwd)/data/convex/cache/identifier_predicates.json $(pwd)/data/convex/
rm data.zip

wget http://qa.mpi-inf.mpg.de/convex/ConvQuestions_test.zip
unzip ConvQuestions_test.zip -d data/convex
rm ConvQuestions_test.zip

#wget http://gaia.infor.uva.es/hdt/wikidata/wikidata2018_09_11.hdt.gz
#unzip wikidata2018_09_11.hdt.gz -d $(pwd)/data/kb
