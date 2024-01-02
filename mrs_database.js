Array.prototype.contains = function(obj) {
    var i = this.length;
    while (i--) {
        if (this[i] === obj) {
            return true;
        }
    }
    return false;
}

var clicked = false

function collapse(form)
{
	let healthy  = parseInt(document.getElementById('N_Healthy'       ).value);
	let clinical = parseInt(document.getElementById('N_Clinical'      ).value);
	let protcols = parseInt(document.getElementById('N_Protocols'     ).value);
	let regions  = parseInt(document.getElementById('N_Regions'       ).value);
	let times    = parseInt(document.getElementById('N_Timepoints'    ).value);
	let treats   = parseInt(document.getElementById('N_Treatments'    ).value);
	let others   = parseInt(document.getElementById('other_conditions').value);
	let total    = ((healthy + clinical) * protcols * regions * times * treats * others);

	const sect_extend = ['subject_info'    ,
						 'acquisition_info', 
						 'analysis_info'   ];

	for (let ss = 0; ss < sect_extend.length; ss++) {
		var all     = document.getElementById(sect_extend[ss]).getElementsByTagName('p');

		for (let ii   = 0; ii < (all.length-1); ii++) {
			var main_id   = document.getElementById('text_' + all[ii].id);
			
			for (let jj = 0; jj < (total); jj++) {
				var target_id = document.getElementById('text_' + all[ii].id + '_' + (jj+1));
				main_id.value   = main_id.value + ';_' + target_id.value;
			}
		}	
	}

	const sect_extend_ = ['values_tcr'      ,
						  'values_iu'       ,
						  'values_mm'       ,
						  'values_t1'       ,
						  'values_t2'       ];

	for (let ss = 0; ss < sect_extend_.length; ss++) {
		var all     = document.getElementById(sect_extend_[ss]).getElementsByTagName('p');

		for (let ii   = 0; ii < (all.length-1); ii++) {			
			if (ii == 0){
				var main_id   = document.getElementById('text_' + all[ii].id);

				for (let jj = 0; jj < (total); jj++) {
					var target_id = document.getElementById('text_' + all[ii].id + '_' + (jj+1));
					main_id.value   = main_id.value + ';_' + target_id.value;
				}
			} 
			else {
				var main_u_id   = document.getElementById('text_' + all[ii].id + '_u');
				var main_s_id   = document.getElementById('text_' + all[ii].id + '_s');

				for (let jj = 0; jj < (total); jj++) {
					var target_u_id = document.getElementById('text_' + all[ii].id + '_u' + '_' + (jj+1));
					var target_s_id = document.getElementById('text_' + all[ii].id + '_s' + '_' + (jj+1));

					if (ii > 1){
						main_u_id.value = main_u_id.value + ';_' + target_u_id.value;
						main_s_id.value = main_s_id.value + ';_' + target_s_id.value;
					}
				}
			}
		}	
	}
	document.getElementById("submission_button").style.display = 'block';
}

function checkmark(me) 
{
	var id_len = me.id.split('_').length
	let id_    = me.id.split('_')[id_len-1]
	if (document.getElementById(me.id).checked) {
		document.getElementById("values_" + id_).style.display = 'block';
	} else {
		document.getElementById("values_" + id_).style.display = 'none';
	}
	checkmark_to_upload()
}

function check_autofill(me) 
{
	var id_len   = me.id.split('_').length
	var id       = me.id.substring(6)

	var text_val = document.getElementById( 'text_' + id + '_1' ).value

	var all      = document.getElementById(id).getElementsByTagName('input')

	for (let ii = 2; ii < all.length; ii++) {
		document.getElementById( 'text_' + id + '_' + ii ).value = text_val;
	}
}

function checkmark_to_upload() 
{	
	if (document.getElementById('values_selection_checkbox_tcr').checked ||
		document.getElementById('values_selection_checkbox_iu' ).checked ||
		document.getElementById('values_selection_checkbox_mm' ).checked ||
		document.getElementById('values_selection_checkbox_t1' ).checked ||
		document.getElementById('values_selection_checkbox_t2' ).checked   ) 
	{
		document.getElementById("submission_notes").style.display = 'block';
	} else {
		document.getElementById("submission_notes").style.display = 'none';
	}
}

function Calculate(form) {
	let first    = 			document.getElementById('submission_first_name' ).value;
	let middle   = 			document.getElementById('submission_middle_name').value;
	let last     = 			document.getElementById('submission_last_name'  ).value;
	let email    =  		document.getElementById('submission_email'      ).value;
	let healthy  = parseInt(document.getElementById('N_Healthy'             ).value);
	let clinical = parseInt(document.getElementById('N_Clinical'            ).value);
	let protcols = parseInt(document.getElementById('N_Protocols'           ).value);
	let regions  = parseInt(document.getElementById('N_Regions'             ).value);
	let times    = parseInt(document.getElementById('N_Timepoints'          ).value);
	let treats   = parseInt(document.getElementById('N_Treatments'          ).value);
	let others   = parseInt(document.getElementById('other_conditions'      ).value);
	let total    = ((healthy + clinical) * protcols * regions * times * treats * others);

	for (let ii = 0; ii < healthy; ii++) {
		for (let jj = 0; jj < protcols; jj++) {
			for (let kk = 0; kk < regions; kk++) {
				for (let ll = 0; ll < times; ll++) {
					for (let mm = 0; mm < treats; mm++) {
						for (let nn = 0; nn < others; nn++) {

						var newRow = document.getElementById('GroupsTable').insertRow();
						newRow.innerHTML = ('<td>Healthy '       + (ii+1) + '</td>' + 
											'<td align = right>' + (jj+1) + '</td>' + 
											'<td align = right>' + (kk+1) + '</td>' + 
											'<td align = right>' + (ll+1) + '</td>' + 
											'<td align = right>' + (mm+1) + '</td>' + 
											'<td align = right>' + (nn+1) + '</td>'   );
						}
					}
				}
			}
		}
	}

	for (let ii = 0; ii < clinical; ii++) {
		for (let jj = 0; jj < protcols; jj++) {
			for (let kk = 0; kk < regions; kk++) {
				for (let ll = 0; ll < times; ll++) {
					for (let mm = 0; mm < treats; mm++) {
						for (let nn = 0; nn < others; nn++) {
				
						var newRow = document.getElementById('GroupsTable').insertRow();
						newRow.innerHTML = ('<td>Clinical '      + (ii+1) + '</td>' + 
											'<td align = right>' + (jj+1) + '</td>' + 
											'<td align = right>' + (kk+1) + '</td>' + 
											'<td align = right>' + (ll+1) + '</td>' + 
											'<td align = right>' + (mm+1) + '</td>' + 
											'<td align = right>' + (nn+1) + '</td>'  );
						}
					}
				}
			}
		}
	}

	const sect_extend = ['subject_info'    ,
						 'acquisition_info', 
						 'analysis_info'   ];
	const sect_incl   = [0]

	for (let ss = 0; ss < sect_extend.length; ss++) {
		var all     = document.getElementById(sect_extend[ss]).getElementsByTagName('p');
		for (let ii = 0; ii < (total-1); ii++) {

			for (let jj = 0; jj < all.length; jj++) {
				let id  = all[jj].id + '_'
				if (jj == 0 && sect_incl.contains(ss)) {
					all[jj].innerHTML = all[jj].innerHTML + '<input type="text" id="text_' + id + (ii+2) + '" ' + 'value="Group ' + (ii+2) + '" ' + 'style="width: 100px;">';
				} else {
					all[jj].innerHTML = all[jj].innerHTML + '<input type="text" id="text_' + id + (ii+2) + '" ' + 									 'style="width: 100px;">';
				}
			}
		}
	}

	const sect_extend_ = ['values_tcr'      ,
						  'values_iu'       ,
						  'values_mm'       ,
						  'values_t1'       ,
						  'values_t2'       ];

	for (let ss = 0; ss < sect_extend_.length; ss++) {
		var all     = document.getElementById(sect_extend_[ss]).getElementsByTagName('p');

		for (let ii = 0; ii < (total-1); ii++) {

			for (let jj = 0; jj < all.length; jj++) {
				let id  = all[jj].id + '_'
				if (jj == 0) {
					all[jj].innerHTML = (all[jj].innerHTML + 
										 '<input type="text" id="text_' + id + (ii+2) + '" value="Group ' + (ii+2) + '" ' + 'style="width: 100px;border: 0;text-align: center;">');
				} else if (jj == 1) {
					all[jj].innerHTML = (all[jj].innerHTML +
										' <input type="text" id="text_' + id + 'l_' + (ii+2) + '" ' + 'style="width:  5px;border: 0;text-align: center;outline: none;" disabled="disabled">' +
										' <input type="text" id="text_' + id + 'u_' + (ii+2) + '" ' + 'style="width: 42px;border: 0;text-align: center;" value="Mean"  disabled="disabled">' +
										' <input type="text" id="text_' + id + 'p_' + (ii+2) + '" ' + 'style="width: 10px;border: 0;text-align: center;" value="&#177" disabled="disabled">' +
										' <input type="text" id="text_' + id + 's_' + (ii+2) + '" ' + 'style="width: 42px;border: 0;text-align: center;" value="Std."  disabled="disabled">' +	
										' <input type="text" id="text_' + id + 'r_' + (ii+2) + '" ' + 'style="width:  5px;border: 0;text-align: center;outline: none;" disabled="disabled">' );
				} else {
					all[jj].innerHTML = (all[jj].innerHTML +
										' <input type="text" id="text_' + id + 'l_' + (ii+2) + '" ' + 'style="width:  5px;border: 0;text-align: center;outline: none;" disabled="disabled">' +
										' <input type="text" id="text_' + id + 'u_' + (ii+2) + '" ' + 'style="width: 42px;">'													  +
										' <input type="text" id="text_' + id + 'p_' + (ii+2) + '" ' + 'style="width: 10px;border: 0;text-align: center;" value="&#177" disabled="disabled">' +
										' <input type="text" id="text_' + id + 's_' + (ii+2) + '" ' + 'style="width: 42px;">' 													  +	
										' <input type="text" id="text_' + id + 'r_' + (ii+2) + '" ' + 'style="width:  5px;border: 0;text-align: center;outline: none;" disabled="disabled">' );
				}
			}
		}
	}
}
