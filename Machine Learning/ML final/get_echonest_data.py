from pyechonest import config
from pyechonest import artist
from pyechonest import song
import time

#config.ECHO_NEST_API_KEY = 'KXNMI1XXUNPX49FQQ'  # Andrew's key
#config.ECHO_NEST_API_KEY = 'M5HXQCWH2IUR5CM8X'  # Max's key
config.ECHO_NEST_API_KEY = 'HKFOXIEQGHVILQNQC'  # Dan's key


# Parameters
labels      = ['electronic', 'metal', 'rap', 'classical', 'reggae', 'rock', 'pop', 'jazz']
#label_i     = 0
label_i    = 6
num_results = 900
idx         = 1
idx_inc     = 100

# File names
svm_file_name   = 'svm_' + labels[label_i] + '.txt'
tag_file_name   = 'tags_' + labels[label_i] + '.txt'

# Initialize SVM and tag files
svm_file = open(svm_file_name, 'w')
tag_file = open(tag_file_name, 'w')
svm_file.write('')
tag_file.write('')
svm_file.close()
tag_file.close()
svm_file = open(svm_file_name, 'a')
tag_file = open(tag_file_name, 'a')

total_songs   = 0
empty_artists = 0

# Loop to avoid EchoNest limit on results count
while (idx < num_results):

    print str(total_songs) + " songs gathered... bout to get " + str(idx_inc) + " more artists..."

    # Query for artists of the given genre
    artist_results = artist.search(style=labels[label_i], results=idx_inc, start=idx)

    # Increment results index
    idx += idx_inc

    # For each artist...
    for a in artist_results:
        try:
            # Get a song for given artist
            result = song.search(artist=a, results=10)
            # If no songs for given artist, skip
            if (len(result) == 0):
                empty_artists += 1
                continue
            for s in result:
                #s = result[0]
                # Get the feature values
                summary  = s.audio_summary
                dance    = summary['danceability']
                dur      = summary['duration']
                energy   = summary['energy']
                loud     = summary['loudness']
                speech   = summary['speechiness']
                acoustic = summary['acousticness']
                live     = summary['liveness']
                tempo    = summary['tempo']
                time_sig = summary['time_signature']
                #hottt    = s.song_hotttnesss
                # Write genre label
                svm_file.write(str(label_i + 1))
                vals = [dance, dur, energy, loud, speech,
                        acoustic, live, tempo, time_sig]
                attr = 1
                # Write feature/value pairs to SVM file
                for val in vals:
                    svm_file.write(' ' + str(attr) + ':' + str(val))
                    attr += 1
                svm_file.write('\n')
                # Write song/artist names to tag file
                title = s.title.encode('utf8')
                name = s.artist_name.encode('utf8')
                tag_file.write(str(title) + ' ||| ' + str(name) + '\n')
                total_songs += 1
                # Sleep to avoid exceeding rate limit
                time.sleep(1.15)
        except Exception:
            continue


# Close files
svm_file.close()
tag_file.close()

# Print results
print "\nNumber of total songs: " + str(total_songs)
print "Number of empty artists: " + str(empty_artists)
